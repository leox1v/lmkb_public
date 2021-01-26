from typing import List, Optional, Dict, Union
from absl import logging
from collections import defaultdict
import json
import os
from tqdm import tqdm
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
import torch


LAMA_DIR = 'data/LAMA'
BATS_DIR = 'data/BATS_3.0'

def load_jsonl(fname: str):
    with open(fname, 'r') as f:
        json_list = list(f)

    results = []
    for json_str in json_list:
        results.append(json.loads(json_str))
    return results

def load_relations():
    relations = load_jsonl(os.path.join(LAMA_DIR, 'relations.jsonl'))
    relations_dict = {x['relation']: x for x in relations}
    return relations, relations_dict

def load_lama_data(identifier: str):
    assert identifier in ['ConceptNet', 'Google_RE', 'Squad', 'TREx'], \
            'Invalid identifier'
    data_dir = os.path.join(LAMA_DIR, identifier)
    data = []
    for fname in tqdm(os.listdir(data_dir), desc=f'Loading {identifier} data'):
        data += load_jsonl(os.path.join(data_dir, fname))

    relations, relations_dict = load_relations()
    return data, relations_dict

def load_data(identifier: str, vocab: Optional[List[str]]):
    if identifier == 'BATS':
        return load_bats(vocab)
    return load_simplified_lama(identifier, vocab)


def load_bats(vocab: Optional[List[str]]):
    def load_bats_file(fname):
        with open(fname, 'r') as f:
            data = [l.strip('\n') for l in f.readlines()]
        return [(r.split('\t')[0], r.split('\t')[1].split('/')) for r in data]
    fnames = [os.path.join(BATS_DIR, d, f) for d in os.listdir(BATS_DIR)
            if not '.' in d for f in os.listdir(os.path.join(BATS_DIR, d))]
    lc_vocab = [t.lower() for t in vocab]
    data = []
    total_solvable = 0
    solvable = defaultdict(int)
    relations_examples = defaultdict(list)
    for fname in tqdm(fnames, desc='Loading BATS data'):
        new_data = load_bats_file(fname)
        rel = fname.split('/')[-1].split()[0]
        rel_data = []
        num_solvable = 0
        for i, (x, y) in enumerate(new_data):
            y = [sol for sol in y if sol in vocab or sol in lc_vocab]
            # if not y:
            #     continue
            example = dict(
                x=x,
                y=y,
                subj=x,
                rel=rel,
                uuid=f'{rel}_{i}',
                other_solutions=[],
            )
            rel_data.append(example)
            if y:
                num_solvable += 1
                relations_examples[rel].append(example)
        if num_solvable < 1:
            logging.warning(f'Removing relation {rel} from the data as only ' \
                    f'{len(rel_data)}/50 objects are in the vocabulary.')
            if rel in relations_examples:
                del relations_examples[rel]
        else:
            data += rel_data
            total_solvable += num_solvable
        solvable[rel] = num_solvable
        # logging.info(f'For relation {rel} {num_solvable/50*100:.2f}% can be solved.')

    logging.info(f'Finished loading BATS data. Kept {len(data)}/2000 examples (but only {total_solvable}/2000 are solvable).')
    with open('data/BATS_3.0/solvable.txt', 'w') as f:
        f.write('\n'.join([f'{k}\t{v}' for k, v in solvable.items()]))
    return data, relations_examples

def load_simplified_lama(identifier: str, vocab: Optional[List[str]]):
    data, relations_dict = load_lama_data(identifier)

    if identifier == 'TREx':
        # Find other possible solutions for N-M relations.
        nm_solutions = defaultdict(list)
        for example in data:
            rel = example['predicate_id']
            _type = relations_dict[rel]['type']
            if  _type == 'N-M':
                sub = example['sub_label']
                y = example['obj_label']
                nm_solutions[f'{rel}-{sub}'].append(y)

    simplified_data = []
    relations_examples = defaultdict(list)

    unique_examples = []
    for example in tqdm(data, desc=f'Processing {identifier} data'):
        if identifier == 'TREx':
            rel = example['predicate_id']
            _type = relations_dict[rel]['type']
            template = relations_dict[rel]['template']
            # Replace [X] placeholder with subject.
            x = template.replace('[X]', example['sub_label'])
            # Replace [Y] placeholder with [MASK].
            x = x.replace('[Y]', '[MASK]')
            y = example['obj_label']
            subj = example['sub_label']
            rel = example['predicate_id']
            other_solutions = []
            if _type == 'N-M':
                # All the possible solutions to this example.
                solutions = nm_solutions[f'{rel}-{subj}']
                other_solutions = list(set(solutions) - set([y]))
        elif identifier == 'ConceptNet':
            # x = example['masked_sentences'][0]
            x = ' '.join(example['masked_sentences'])
            y = example['obj_label']
            subj = example['sub'] if 'sub' in example else example['sub_label']
            rel = example['pred']
            other_solutions = []
        elif identifier == 'Google_RE':
            # x = ' '.join(example['masked_sentences'])
            y = example['obj_label']
            subj = example['sub_label']
            rel = example['pred']
            x = {
                'place_of_birth': f'{subj} was born in [MASK].',
                'date_of_birth': f'{subj} (born [MASK]).',
                'place_of_death': f'{subj} died in [MASK].',
            }[rel.split('/')[-1]]
            other_solutions = []
        else:
            raise NotImplementedError

        # Filter out duplicates.
        if (rel, subj, y) in unique_examples:
            continue
        unique_examples.append((rel, subj, y))

        simplified_example = {
            'x': x,
            'y': y,
            'subj': subj,
            'rel': rel,
            'uuid': example['uuid'],
            'other_solutions': other_solutions,
        }

        def is_valid_example(example: Dict[str, str]) -> bool:
            if not y or not subj:
                return False
            max_sentence_length = 100
            if len(x.split()) > max_sentence_length:
                return False
            if y not in vocab:
                return False
            return True

        if is_valid_example(simplified_example):
            simplified_data.append(simplified_example)
            relations_examples[rel].append(simplified_example)

    logging.info(f'Finished loading {identifier} data. Kept {len(simplified_data)}/{len(data)} examples.')
    return simplified_data, relations_examples

def load_vocab(fname: str):
    with open(fname, 'r') as f:
        vocab = [token.strip('\n') for token in f.readlines()]
    return vocab

class PrimingDataLoader:
    def __init__(self, batch_size: int, identifier: str,
            num_priming_examples: int = 20,
            priming_type: str='nl',
            use_close_examples: bool=True,
            random_seed: int=0,
            vocab: Union[str, List[str]]=''):


        if isinstance(vocab, str):
            vocab = load_vocab(vocab) if vocab else None

        self.data, self.relations_examples = load_data(identifier,
                vocab=vocab)
        self.batch_size = batch_size
        self.num_priming_examples = num_priming_examples
        self.priming_type = priming_type
        self.use_close_examples = use_close_examples
        np.random.seed(random_seed)
        if use_close_examples:
            try:
                with open(f'data/related_relation_examples_{identifier}.pickle', 'rb') as handle:
                    self.related_relation_examples = pickle.load(handle)
            except:
                logging.info('Start encoding the subjects.')
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                model = AutoModel.from_pretrained('bert-base-uncased').to(device)
                subjects_per_relation = {k: list(set([e['subj'] for e in v]))
                                      for k, v in self.relations_examples.items()}
                all_subjects = list(set([s for v in subjects_per_relation.values() for s in v]))
                subject_encodings = []
                batch_size = 512
                for i in tqdm(range(len(all_subjects)//batch_size+1)):
                    batch = all_subjects[i*batch_size: (i+1)*batch_size]
                    if not batch:
                        break
                    inputs = tokenizer(batch, return_tensors='pt', padding=True).to(device)
                    with torch.no_grad():
                        out = model(**inputs).last_hidden_state[:, 0, :]
                    subject_encodings.append(out)

                subject_encodings = torch.cat(subject_encodings)
                similarity_matrix = torch.matmul(subject_encodings, subject_encodings.T) / torch.norm(subject_encodings, dim=1)**2
                _, most_similar_idxs = torch.topk(similarity_matrix, k=min(301, len(subject_encodings)), dim=1)
                most_similar_idxs = most_similar_idxs.cpu().numpy()

                def find(arr, val, not_found_idx=400):
                    res = np.where(arr == val)[0]
                    if len(res) == 0:
                        return not_found_idx
                    return res[0]

                self.related_relation_examples = defaultdict(dict)
                for relation, subjects in tqdm(subjects_per_relation.items()):
                    subject_idxs = [all_subjects.index(s) for s in subjects]
                    subject_to_index = {s: i for s, i in zip(subjects, subject_idxs)}
                    for subject, idx in subject_to_index.items():
                        sorted_subject_ids = sorted(subject_idxs, key=lambda i: find(most_similar_idxs[idx], i) if i != idx else 1000) # to remove self similarity.
                        sorted_subjects = [all_subjects[i] for i in sorted_subject_ids[:20]]
                        # Choose only examples that are in the top10 most related.
                        examples = [e for e in self.relations_examples[relation] if e['subj'] in sorted_subjects]
                        self.related_relation_examples[relation][subject] = examples

                logging.info('Finished encoding the subjects.')
                with open(f'data/related_relation_examples_{identifier}.pickle', 'wb') as handle:
                    pickle.dump(self.related_relation_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def __len__(self):
        return int(len(self.data)/self.batch_size)

    def __iter__(self):
        return self._example_generator(self.data)

    def _example_generator(self, data):
        data = np.random.permutation(data)
        i, total_steps = 0, len(data) // self.batch_size
        for i in range(total_steps + 1):
            examples = data[i * self.batch_size: (i+1) * self.batch_size]
            if len(examples) == 0:
                break

            examples_dict = defaultdict(list)
            # Transformation.
            # + Make it a dictionary with lists as values.
            for e in examples:
                e['x_primed'] = self.get_priming(example=e,
                        k = self.num_priming_examples,
                        priming_type=self.priming_type)
                for k, v in e.items():
                    examples_dict[k].append(v)

            yield examples_dict

    def get_priming(self, example, k: int=20,
            priming_type: str='nl') -> str:
        relation_id = example['rel']
        if self.use_close_examples:
            priming_examples = self.related_relation_examples[relation_id][example['subj']]
        else:
            priming_examples = self.relations_examples[relation_id]


        priming_examples = [e for e in priming_examples
                            if e['subj'] != example['subj']]
        priming_examples = np.random.choice(priming_examples,
                min(k, len(priming_examples)),
                replace=True)
        if priming_type == 'nl':
            priming = []
            for e in priming_examples:
                assert '[MASK]' in e['x'], f'No mask provided in the input \
                        "{e["x"]}".'
                priming.append(e['x'].replace('[MASK]', e['y']))
            priming += [example['x']]
        else:
            priming = []
            for e in priming_examples:
                y = e['y'] if isinstance(e['y'], str) else \
                        np.random.choice(e['y'])
                priming.append(priming_type.format(e['subj'], y))
            priming += [priming_type.format(example['subj'], '[MASK]') + '\n']
        priming_str = '\n'.join(priming)
        return priming_str




