from typing import List, Dict
import torch
from absl import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel
from collections import defaultdict

class LM:
    def __init__(self, name: str, use_common_vocab: bool=True):
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        self.blacklist_mask = torch.zeros(max(self.tokenizer.vocab.values()) + 1)

        if use_common_vocab:
            # TODO: give the file path as argument.
            fname = 'vocab/common_vocab_cased.txt'
            with open(fname, 'r') as f:
                common_vocab = [token.strip('\n') for token in f.readlines()]

            logging.info(f'Using common vocab of size {len(common_vocab)}'
                         f' loaded from {fname}.')

            if 'albert' in name or '-uncased' in name:
                common_vocab = [t.lower() for t in common_vocab]

            vocab_dict = {self.tokenizer.convert_tokens_to_string([t]): idx
                    for t, idx in self.tokenizer.get_vocab().items()}
            for k, v in vocab_dict.items():
                if k not in common_vocab:
                    self.blacklist_mask[v] = 1.

        if 't5' in name.lower():
            self.model = T5ForConditionalGeneration.from_pretrained(name, return_dict=True)
            self.tokenizer.mask_token = '<extra_id_0>'
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(name,
                    return_dict=True)
        self.model.eval()

    @property
    def is_mask_based(self):
        return self.tokenizer.mask_token_id is not None

    @property
    def is_generation_model(self):
        return any([_id in self.tokenizer.name_or_path.lower()
            for _id in ['t5', 'gpt']])

    @property
    def device(self):
        return str(next(self.model.parameters()).device)

    @property
    def vocab(self) -> List[str]:
        return [self.tokenizer.convert_tokens_to_string([token])
                for token in self.tokenizer.vocab.keys()]

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def preprocess_sentence(self, sentence: str) -> str:
        if not self.is_mask_based:
            # For non-masked models like GPT-2.
            return sentence.split('[MASK]')[0].strip()
        assert '[MASK]' in sentence, \
                f'No masking token provided in sentence {sentence}.'
        return sentence.replace('[MASK]', self.tokenizer.mask_token)

    def mask_token_occurence_idx(self,
            input_ids: torch.LongTensor) -> torch.LongTensor:
        if self.tokenizer.mask_token_id is None:
            # Model has no mask token, e.g. GPT-2, then just return the last idx
            return torch.ones(input_ids.size(0)) * input_ids.size(1)
        return (input_ids == self.tokenizer.mask_token_id).long().argmax(-1)

    def prepare_input(self, texts: List[str]) -> torch.LongTensor:
        x = list(map(self.preprocess_sentence, texts))
        # Hack that we need to do to cut from the left because 'truncate from left'
        # is not yet supported by transformers library but feature has been requested:
        # https://github.com/huggingface/transformers/issues/4476).
        x = [self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(xi)[-500:]) for xi in x]
        inputs = self.tokenizer(x, return_tensors='pt', padding=True)
        return inputs.to(self.device)

    def __call__(self, *args, **kwargs):
        return self.predict_token(*args, **kwargs)

    def generate_token(self, inputs: Dict[str, torch.LongTensor], k: int=10,
            max_length=5, num_beams=10, early_stopping=True):
        """
        The same as predict_token but for generation models like T5 or GPT.
        Args:
            inputs: Input LongTensors for 'input_ids', 'token_type_ids',
              and 'attention_mask'.
            k: The maximum number of predicted tokens.
        """
        k=1
        max_length = 2 + 2
        with torch.no_grad():
            outputs = self.model.generate(**inputs,
                    max_length=max_length,
                    # num_beams=max(num_beams, k),
                    num_beams=1,
                    early_stopping=early_stopping,
                    num_return_sequences=k,
                    # no_repeat_ngram_size=1,
                    # do_sample=True,
                    # top_p=0.9,
                    # top_k=0,
                    # length_penalty=-.1, # https://github.com/huggingface/transformers/blob/master/src/transformers/generation_beam_search.py
                    # For higher values (>1.) it favors longer sequences while for lower values (< 1.0) it favors shorter ones.
            )
        y_pred = [[]]
        y_pred_prob = [[]]
        for o in outputs:
            prediction = self.tokenizer.decode(o, skip_special_tokens=True).split('.')[0].strip()
            if len(y_pred[-1]) >= k:
                y_pred.append([])
                y_pred_prob.append([])
            y_pred[-1].append(prediction)
            y_pred_prob[-1].append(1. if len(y_pred_prob) == 0 else 0.)

        return dict(
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
        )

    def predict_token(self, inputs: Dict[str, torch.LongTensor], k: int=10,
            save_cls_embedding: bool=False):
        """
        Args:
            inputs: Input LongTensors for 'input_ids', 'token_type_ids',
              and 'attention_mask'.
            k: The maximum number of predicted tokens.
        """
        self.model.eval()
        if self.is_generation_model:
            return self.generate_token(inputs, k)

        bs = inputs.input_ids.size(0)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=save_cls_embedding)

        cls_embedding = None
        if save_cls_embedding:
            # Embedding of shape [batch_size, hidden_dim]
            cls_embedding = outputs.hidden_states[-1][:, 0, :]
            cls_embedding = cls_embedding.tolist()
            # cls_embedding = [arr for arr in cls_embedding.cpu().detach().numpy()]

        mask_idxs = self.mask_token_occurence_idx(inputs['input_ids'])
        mask_logits = outputs.logits[torch.arange(mask_idxs.size(0)).to(self.device),
                                     mask_idxs]
        # Set the logits according to the blacklisted words to very small value.
        mask_logits = mask_logits - (1e20) * self.blacklist_mask.repeat(bs, 1).to(self.device)
        _, topk_indices = torch.topk(mask_logits, k)

        probs = torch.softmax(mask_logits, -1).gather(1, topk_indices)

        prediction_tokens = [self.tokenizer.convert_ids_to_tokens(idx)
                for idx in topk_indices]
        prediction = [self.tokenizer.convert_tokens_to_string(tokens).strip().split()
                for tokens in prediction_tokens]

        sorted_pred = [sorted(
            [(token, p.item()) for token, p in zip(tokens, token_probs)],
            key=lambda x: x[1], reverse=True)
            for tokens, token_probs in zip(prediction, probs)]

        out = dict(
                y_pred=[[token[0] for token in token_lists] for token_lists in sorted_pred],
                y_pred_prob=[[token[1] for token in token_lists] for token_lists in sorted_pred],
                cls_embedding=cls_embedding,
        )

        return out
