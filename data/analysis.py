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
        target = row.y.lower()
        predictions = [p.lower() for p in row.y_pred]
        predictions_probabilities = row.y_pred_prob
        # if 'other_solutions' in row and len(set(predictions) - set(row.other_solutions)) > 0:
        #     predictions, predictions_probabilities = zip(*[
        #         (y, prob) for y, prob in zip(predictions, predictions_probabilities)
        #         if y not in row.other_solutions
        #     ])

        # Compute Precision.
        precisions = {f'p@{k}': target in predictions[:k] for k in precision_ks}

        # Compute MRR.
        mrr = 1. / (predictions.index(target) + 1) if target in predictions else 0.

        # Compute the likelihood of the correct answer.
        likelihood = predictions_probabilities[predictions.index(target)] \
                if target in predictions else 0

        return dict(
            **precisions,
            mrr=mrr,
            target_likelihood=likelihood,
        )

    res = df.apply(lambda row: compute_metrics(row, ks), axis=1)
    return pd.DataFrame(list(res)).mean().to_dict()

def process_df_per_type(df: pd.DataFrame, data: str, model: str,
        num_examples: int, use_natural_language: bool,
        use_close_examples: bool, random_seed: int):

    metrics = process_df_per_relation(df)
    metrics_df = pd.DataFrame(metrics)
    reltypes_to_rel = defaultdict(list)

    info_dict = dict(
        model=model,
        num_examples=num_examples,
        use_natural_language=use_natural_language,
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
    for col in [c for c in df.columns if c.startswith('y_pred')]:
        df.loc[:, col] = df[col].apply(literal_eval)

    return process_df_per_type(
            df=df,
            data=data,
            model=model,
            num_examples=int(infos['pe']),
            use_natural_language=infos['nl'] == 'True',
            use_close_examples=infos['ce'] == 'True',
            random_seed=int(infos['rs']))

def print_improvement_examples():
    all_files = get_all_files(FLAGS.results_dir)
    df_options = [
            {'pe': '0', 'nl': 'True', 'ce': 'True', 'rs': '0'},
            {'pe': '1', 'nl': 'True', 'ce': 'False', 'rs': '0'},
            {'pe': '1', 'nl': 'True', 'ce': 'True', 'rs': '0'},
            {'pe': '3', 'nl': 'False', 'ce': 'False', 'rs': '0'},
    ]
    ddata = 'TREx'
    mmodel = 'bert-large-cased'
    dfs = [None] * len(df_options)

    for fname in tqdm(all_files):
        path = fname.replace(FLAGS.results_dir, '').strip('/')
        data = path.split('/')[0]
        model = path.split('/')[1]
        infos = {x.split('-')[0]: x.split('-')[1]
                for x in path.split('/')[2].strip('.csv').split('_')}
        if not (data == ddata and model == mmodel and infos in df_options):
            continue
        df = pd.read_csv(fname)

        # Turn the list predictions into actual lists.
        for col in [c for c in df.columns if c.startswith('y_pred')]:
            df.loc[:, col] = df[col].apply(literal_eval)

        dfsumm = process_df_per_type(
                df=df,
                data=data,
                model=model,
                num_examples=int(infos['pe']),
                use_natural_language=infos['nl'] == 'True',
                use_close_examples=infos['ce'] == 'True',
                random_seed=int(infos['rs']))
        dfs[df_options.index(infos)] = df

    selection = dfs[0].apply(lambda row:
            row.y not in row.y_pred[:3],
            axis=1)
    selection &= dfs[1].apply(lambda row:
            row.y in row.y_pred[:2] and row.y not in row.y_pred[:1],
            axis=1)
    selection &= dfs[2].apply(lambda row:
            row.y in row.y_pred[:1],
            axis=1)
    selection &= dfs[3].apply(lambda row:
            row.y in row.y_pred[:1],
            axis=1)

    # selection &= dfs[0].apply(lambda row:
    #         'plays in' in row.x,
    #         axis=1)

    dfs_pruned = [d[selection] for d in dfs]
    for ((_, r0), (_, r1), (_, r2), (_, r3)) in zip(*[df.iterrows() for df in dfs_pruned]):
        print(f'Q: {r0.x}\tA: {r0.y}')
        print('------')
        for r in [r0, r1, r2, r3]:
            print(r.x_primed)
            print('\t'.join([f'{y} ({yp*100:.1f}%)' for y, yp in zip(r.y_pred[:3], r.y_pred_prob[:3])]))
            print('--')
        print('----------------------------------------------\n')

    import pdb; pdb.set_trace()

def best_relations_tex_table():
    all_files = get_all_files(FLAGS.results_dir)
    df_options = [
            {'pe': '0', 'nl': 'True', 'ce': 'True', 'rs': '0'},
            {'pe': '1', 'nl': 'True', 'ce': 'True', 'rs': '0'},
            {'pe': '3', 'nl': 'True', 'ce': 'True', 'rs': '0'},
            {'pe': '10', 'nl': 'True', 'ce': 'True', 'rs': '0'},
    ]
    ddata = 'TREx'
    mmodel = 'bert-large-cased'
    dfs = [None] * len(df_options)

    for fname in tqdm(all_files):
        path = fname.replace(FLAGS.results_dir, '').strip('/')
        data = path.split('/')[0]
        model = path.split('/')[1]
        infos = {x.split('-')[0]: x.split('-')[1]
                for x in path.split('/')[2].strip('.csv').split('_')}
        if not (data == ddata and model == mmodel and infos in df_options):
            continue
        df = pd.read_csv(fname)

        # Turn the list predictions into actual lists.
        for col in [c for c in df.columns if c.startswith('y_pred')]:
            df.loc[:, col] = df[col].apply(literal_eval)

        dfs[df_options.index(infos)] = process_df_per_relation(df)
    relations = get_trex_relations()
    res = [{rel['relation']: rel['p@1'] for rel in df} for df in dfs]
    gains = sorted([(k, relations[k]['template'], res[1][k] - res[0][k], res[2][k] - res[0][k], res[3][k] - res[0][k]) for k in res[0].keys()], key=lambda x: x[-1], reverse=True)

    lines = []
    for x in gains:
        lines.append(f'{x[0]} & {x[1].replace("[X]", "[s]").replace("[Y]", "[o]")} & {x[2]*100:.1f} & {x[3]*100:.1f} & {x[4]*100:.1f} \\\\')
    tex_str = '\n'.join(lines)
    print(tex_str)

def main(argv):
    best_relations_tex_table()

if __name__ == "__main__":
    app.run(main)
