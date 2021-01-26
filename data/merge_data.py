from typing import List, Dict, Optional, Any
from collections import defaultdict
import json
import pandas as pd
import numpy as np
import os
from absl import app, logging, flags
from ast import literal_eval
from tqdm import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string('results_dir', 'out/res',
        'Directory with the results to merge.')
flags.DEFINE_string('output_file', 'out/summary/df.csv',
        'Output file to save the summary.')


def get_trex_relations():
    lama_dir = 'data/LAMA'
    with open(os.path.join(lama_dir, 'relations.jsonl'), 'r') as f:
        relations = {r['relation']: r for r in [json.loads(json_str) for json_str in list(f)]}
    return relations

def get_all_files(root):
    all_files = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            all_files.append(os.path.join(path, name))
    return all_files


def process_df_per_relation(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [dict(relation=relation, **process_df(df[df.rel == relation]))
            for relation in df.rel.unique()]

def process_df(df: pd.DataFrame, ks: Optional[List[int]]=None) -> Dict[str, float]:
    if not ks:
        ks = [1, 3, 5, 10, 20]

    def compute_metrics(row, precision_ks):
        if isinstance(row.y, str):
            target = [row.y.lower()]
        else:
            target = [t.lower() for t in row.y]
        # target = row.y.lower()
        predictions = [p.lower() for p in row.y_pred]
        predictions_probabilities = row.y_pred_prob
        # if 'other_solutions' in row and len(set(predictions) - set(row.other_solutions)) > 0:
        #     predictions, predictions_probabilities = zip(*[
        #         (y, prob) for y, prob in zip(predictions, predictions_probabilities)
        #         if y not in row.other_solutions
        #     ])

        # Compute Precision.
        precisions = {f'p@{k}': any(t in predictions[:k] for t in target) for k in precision_ks}

        # Compute MRR.
        # Compute the likelihood of the correct answer.
        mrr = 0.
        likelihood = 0.
        if any([t in predictions for t in target]):
            mrr = max(1. / (predictions.index(t) + 1) for t in target if t in predictions)
            likelihood = max(predictions_probabilities[predictions.index(t)] \
                    for t in target if t in predictions)

        return dict(
            **precisions,
            mrr=mrr,
            target_likelihood=likelihood,
        )

    res = df.apply(lambda row: compute_metrics(row, ks), axis=1)
    return pd.DataFrame(list(res)).mean().to_dict()

def process_df_per_type(df: pd.DataFrame, data: str, model: str,
        num_examples: int, priming_type: bool,
        use_close_examples: bool, random_seed: int):

    metrics = process_df_per_relation(df)
    metrics_df = pd.DataFrame(metrics)
    reltypes_to_rel = defaultdict(list)

    info_dict = dict(
        model=model,
        num_examples=num_examples,
        priming_type=priming_type,
        use_close_examples=use_close_examples,
        random_seed=random_seed,
    )

    result_dicts = {}

    if data == 'TREx':
        # Slice the df for the different relations.
        relations = get_trex_relations()
        for r in relations.values():
            reltypes_to_rel[r['type']].append(r['relation'])

    elif data == 'Google_RE':
        metrics_df.loc[:, 'relation'] = metrics_df.apply(lambda row:
                row.relation.split('/')[-1].replace('_', '-'), axis=1)
        reltypes_to_rel = {k: k for k in metrics_df.relation.unique()}

    elif data == 'BATS':
        reltypes_to_rel = {k: k for k in metrics_df.relation.unique()}

    result_dicts[f'{data}'] = {**metrics_df.mean().to_dict(), **info_dict}

    for k, v in reltypes_to_rel.items():
        result_dicts[f'{data}_{k}'] = {**metrics_df[metrics_df.apply(
            lambda row: row.relation in v,
            axis=1,
        )].mean().to_dict(), **info_dict}

    res = pd.DataFrame(result_dicts).T.reset_index().rename(columns={'index': 'data'})
    return res

def summarize_file(fname: str) -> pd.DataFrame:
    path = fname.replace(FLAGS.results_dir, '').strip('/')
    data = path.split('/')[0]
    model = path.split('/')[1]
    infos = {x.split('-')[0]: x.split('-')[1]
            for x in path.split('/')[2].strip('.csv').split('_')}
    df = pd.read_csv(fname)

    # Turn the list predictions into actual lists.
    for col in [c for c in df.columns if c.startswith('y')]:
        if '[' in df.loc[0, col]:
            df.loc[:, col] = df[col].apply(literal_eval)

    priming_type = {
            'True': 'nl',
            'nl': 'nl',
            'False': 'rarr',
            'rarr': 'rarr',
            'Rarr': 'Rarr',
            'par': 'par',
    }[infos['nl']]

    df = process_df_per_type(
            df=df,
            data=data,
            model=model,
            num_examples=int(infos['pe']),
            priming_type=priming_type,
            use_close_examples=infos['ce'] == 'True',
            random_seed=int(infos['rs']))
    return df


def save_append(df: pd.DataFrame):
    if os.path.isfile(FLAGS.output_file):
        index_columns = ['data', 'model', 'num_examples',
                'priming_type', 'use_close_examples', 'random_seed']
        df_old = pd.read_csv(FLAGS.output_file)
        df = pd.concat([df_old, df])
        df.drop_duplicates(subset=index_columns,
                keep='last',
                inplace=True,
                ignore_index=True)

    df.to_csv(FLAGS.output_file,
            header=True,
            index=False)

def main(argv):
    all_files = get_all_files(FLAGS.results_dir)

    for fname in tqdm(all_files):
        df = summarize_file(fname)
        save_append(df)

if __name__ == "__main__":
    app.run(main)
