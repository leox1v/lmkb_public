import numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from abc import ABC, abstractmethod
import json


class Priming(ABC):
    @property
    @abstractmethod
    def priming_examples(self) -> List[Tuple[str, str]]:
        pass

    def generate_priming(self, examples):
        return '\n'.join([self.template.format(*e) for e in examples])

    def generate_input(self, obj, shuffle=True, n=100):
        examples = np.random.permutation(self.priming_examples) if shuffle else self.priming_examples
        examples = examples[:n]
        priming_str = self.generate_priming(examples)
        if not priming_str:
            return self.template.format(obj, '').strip()
        return priming_str + '\n' + self.template.format(obj, '').strip()

class AdaptiveObjLocPriming(Priming):
    def __init__(self, objects, containers, n=100, template='{} -> {}'):
        super().__init__()
        assert len(template.split('{}')) == 3, 'The template requires exactly two "{}"'
        self.template = template
        self.n = n
        with open('games/twc_objects.json') as json_file:
            self.all_objects = {k: v for k, v in json.load(json_file).items()
                                if v['type'] == 'o' or v['type'] == 'f'}

        # Remove all objects present in current scene.
        priming_objects = {k: v for k, v in self.all_objects.items() if k not in objects}

        # Use only examples for the current set of containers.
        priming_objects = {k: v for k, v in priming_objects.items()
                            if len(set(v['locations']).intersection(containers)) > 0}

        self.all_priming_examples = [(k, np.random.choice(list(set(v['locations']).intersection(containers))))
                                     for k, v in priming_objects.items()]

    @property
    def priming_examples(self) -> List[Tuple[str, str]]:
        try:
            return [self.all_priming_examples[i]
                    for i in np.random.choice(len(self.all_priming_examples),
                                              size=min(self.n, len(self.all_priming_examples)))]
        except:
            return [('butter', 'fridge'),]


class LM:
    def __init__(self, name: str):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForMaskedLM.from_pretrained(name, return_dict=True)
        self.vocab_embeddings = None

    @property
    def is_mask_based_model(self):
        return self.tokenizer.mask_token_id is not None

    @property
    def device(self):
        return str(next(self.model.parameters()).device)

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def preprocess_sentence(self, sentence: str) -> str:
        return sentence

    def continuation_prob(self, sentence: str, candidates: List[str]):
        def get_probs(x):
            inputs = self.tokenizer(sentence, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[:, inputs['input_ids'].size(1)-1]
            return torch.softmax(logits, -1).squeeze(0)

        # The candidates need to start with a space to work with GPT-2.
        candidates_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(f' {c}')) for c in candidates]
        candidate_probabilities = [[] for _ in range(len(candidates))]
        model_outputs = {}

        for candidate_idx, candidate_subword_ids in enumerate(candidates_ids):
            for i, _id in enumerate(candidate_subword_ids):
                previously_predicted_ids = candidate_subword_ids[:i]
                identifier = ' '.join(map(lambda x: str(x), previously_predicted_ids))
                x_parts = [sentence, 
                           self.tokenizer.decode(previously_predicted_ids), 
                           self.tokenizer.mask_token, 
                           self.tokenizer.decode(candidate_subword_ids[i+1:]),
                ]
                x_parts = list(filter(lambda x: len(x) > 0, x_parts))
                x_new = ' '.join(x_parts) + '\n'
                if identifier in model_outputs:
                    probs = model_outputs[identifier]
                else:
                    probs = get_probs(x_new)
                    model_outputs[identifier] = probs
                candidate_probabilities[candidate_idx].append(probs[_id].item())

        candidate_probabilities = list(map(lambda x: max(x), candidate_probabilities))
        candidate_probabilities = list(map(lambda x: x/sum(candidate_probabilities), candidate_probabilities))

        return dict(
            words=candidates,
            probs=candidate_probabilities,
        )
