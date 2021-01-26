from typing import List
import pandas as pd
import os
from absl import logging
import numpy as np
import matplotlib as mpl

def generate_results_table(df: pd.DataFrame, output_dir: str):
    # Add models.
    models = [
        (f'Bb$^3$', df[(df.model == 'bert-base-cased') & (df.num_examples == 3) & \
            (df.use_natural_language == True) & (df.use_close_examples == False)]),
        (f'Bb$^{{10}}$', df[(df.model == 'bert-base-cased') & (df.num_examples == 10) & \
            (df.use_natural_language == True) & (df.use_close_examples == False)]),
        (f'Bb$^{{10}}_{{ce}}$', df[(df.model == 'bert-base-cased') & (df.num_examples == 10) & \
            (df.use_natural_language == True) & (df.use_close_examples == True)]),
        (f'Bl$^{{3}}$', df[(df.model == 'bert-large-cased') & (df.num_examples == 3) & \
            (df.use_natural_language == True) & (df.use_close_examples == False)]),
        (f'Bl$^{{10}}$', df[(df.model == 'bert-large-cased') & (df.num_examples == 10) & \
            (df.use_natural_language == True) & (df.use_close_examples == False)]),
        (f'Bl$^{{10}}_{{ce}}$', df[(df.model == 'bert-large-cased') & (df.num_examples == 10) & \
            (df.use_natural_language == True) & (df.use_close_examples == True)]),
        (f'Al$^{{10}}_{{ce}}$', df[(df.model == 'albert-xxlarge-v2') & (df.num_examples == 10) & \
            (df.use_natural_language == True) & (df.use_close_examples == True)]),
    ]
    # Add baseline models.
    baselines = [
        ('Bb', df[(df.model == 'bert-base-cased') & (df.num_examples == 0) & \
            (df.use_natural_language == True) & (df.use_close_examples == False) & \
            (df.random_seed == 0)]),
        ('Bl', df[(df.model == 'bert-large-cased') & (df.num_examples == 0) & \
            (df.use_natural_language == True) & (df.use_close_examples == False) & \
            (df.random_seed == 0)]),
        ('Al', df[(df.model == 'albert-xxlarge-v2') & (df.num_examples == 0) & \
            (df.use_natural_language == True) & (df.use_close_examples == True) & \
            (df.random_seed == 0)]),
        ('Bb$_{{opt}}$', {('Google_RE', 'total'): 0.104,
            ('TREx', 'total'): 0.396}),
        ('Bl$_{{opt}}$', {('Google_RE', 'total'): 0.113,
            ('TREx', 'total'): 0.439}),
    ]

    def av(models: List, model_idx: int, data: str, rel_type: str, key='p@1'):
        # Access values.
        df = models[model_idx][1]

        std = -1
        if isinstance(df, dict):
            value = df.get((data, rel_type), -1)
            if value < 0:
                return -1, -1
        else:
            data_str = f'{data}_{rel_type}' if rel_type != 'total' else data
            df = df[df.data == data_str]
            if len(df) == 0:
                return -1, -1
            # if len(df) > 1:
            #     import pdb; pdb.set_trace()
            value = df[key].mean()
            if len(df) > 1:
                std = df[key].std()

            # value = df.iloc[0][key]
        return value, std
        # return f'{value*100:.1f}'

    def stringify_arr(arr):
        maxv = max([v for v, _ in arr])
        minv = min([v for v, _ in arr if v > 0])
        norm = mpl.colors.Normalize(vmin=minv, vmax=1.6*maxv)
        cmap = mpl.cm.get_cmap('YlGn')
        values_str = []
        for v, std in arr:
            if v < 0:
                v_str = '-'
            else:
                rgb_color = cmap(norm(v))[:3]
                v_str = f'{v*100:.1f}'
                if v == maxv:
                    v_str = f'\\textbf{{{v_str}}}'
                if std > 0.:
                    std *= 100
                    std_str = f'{std:.1f}'# .replace('0.', '.')
                    v_str += f' $_{{\\pm {std_str}}}$'
                # v_str = f'${v_str}$'
                v_str = f'\\cellcolor[rgb]{{{", ".join([str(v) for v in rgb_color])}}} {v_str}'
            values_str.append(v_str)
        return values_str

    all_models = baselines + models

    values = []
    for data in ['Google_RE', 'TREx', 'ConceptNet']:
        relations = {'Google_RE': ['place-of-birth', 'date-of-birth', 'place-of-death', 'total'],
                'TREx': ['1-1', 'N-1', 'N-M', 'total'],
                'ConceptNet': ['total']}[data]
        for rel in relations:
            values.append([av(all_models, idx, data, rel) for idx in range(len(all_models))])

    values = [stringify_arr(v) for v in values]

    table_str = f"""
    \\begin{{tabular}}{{ll{'c'*len(all_models)}}}
    \\toprule
    \\multirow{{2}}{{*}}{{Corpus}} & \\multirow{{2}}{{*}}{{Relation}} & \\multicolumn{{{len(baselines)}}}{{|c}}{{Baselines}} & \\multicolumn{{{len(models)}}}{{|c}}{{LM}}  \\\\
             & & {' & '.join([m[0] for m in all_models])}  \\\\
    \\midrule
    \\multirow{{4}}{{*}}{{Google-RE}}  & \\rel{{birth-place}} & {' & '.join(values[0])} \\\\
    &\\rel{{birth-date}} & {' & '.join(values[1])} \\\\
    &\\rel{{death-place}} & {' & '.join(values[2])} \\\\
    \\cmidrule{{2-14}}
    & Total & {' & '.join(values[3])} \\\\
    \\midrule
    \\multirow{{4}}{{*}}{{T-REx}}  & $1$-$1$ & {' & '.join(values[4])} \\\\
    &$N$-$1$ &  {' & '.join(values[5])} \\\\
    & $N$-$M$ &  {' & '.join(values[6])} \\\\
    \\cmidrule{{2-14}} 
    & Total & {' & '.join(values[7])} \\\\
    \\midrule 
    ConceptNet & Total & {' & '.join(values[8])} \\\\
    \\bottomrule
    \\end{{tabular}}
    """

    fname = os.path.join(output_dir, 'results_table.tex')
    with open(fname, 'wt') as file:
        file.write(table_str)
    logging.info(f'Saved results table to {fname}.')
