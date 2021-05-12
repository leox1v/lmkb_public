import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_results(fname):
    with open(fname, 'r') as f:
        results = []
        for line in f.readlines():
            line = line.strip('\n')
            example_dict = {}
            for example in line.split(','):
                k, v = example.split(':')
                if k == 'num_priming_examples':
                    v = int(v)
                elif k == 'score' or k == 'steps':
                    v = float(v)
                example_dict[k] = v
            results.append(example_dict)
    df = pd.DataFrame.from_dict(results)
    return df

def plot(df):
    sns.set(font_scale=1.4)
    plt.style.use(['seaborn-whitegrid'])
    colors = sns.color_palette()

    def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, **kwargs):
        ax = ax if ax is not None else plt.gca()
        x, y, yerr = np.array(x), np.array(y), np.array(yerr)
        if color is None:
            color = next(ax._get_lines.prop_cycler)['color']
        if np.isscalar(yerr) or len(yerr) == len(y):
            ymin = y - yerr
            ymax = y + yerr
        elif len(yerr) == 2:
            ymin, ymax = yerr
        ax.plot(x, y, '--', color=color, **kwargs)
        ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

    def _change_legend(legend, mapping, title):
        for t in legend.texts:
            if t.get_text() in mapping:
                t.set_text(mapping[t.get_text()])
        legend.set_title(title)

    def _save(fig, fname: str, base_dir: str = ''):
        fname = os.path.join(base_dir, f'{fname}.pdf')
        print(f'Saving plot {fname}.')
        fig.savefig(fname,
                    format='pdf',
                    bbox_inches='tight')

    filter_fn = lambda \
        row: row.priming_template == 'arrow' and row.agent == 'lmkb' and 'hard' in row.games and row.num_priming_examples <= 20
    df_current = df[df.apply(filter_fn, axis=1)]
    fig = plt.figure()

    # Add baseline
    xx = sorted(df_current.num_priming_examples.unique())
    uniform_baseline_scores = df[df.apply(lambda row: row.agent == 'lmkb-random' and 'hard' in row.games, axis=1)].score
    baselines = {
        'TWC': (0.57, 0.),
        'uniform prior': (np.mean(uniform_baseline_scores), np.std(uniform_baseline_scores)),
    }
    for i, (k, v) in enumerate(baselines.items()):
        errorfill(xx, [v[0]] * len(xx), yerr=[v[1]] * len(xx), label=k, color=colors[i])
    g = sns.lineplot(data=df_current, x='num_priming_examples', y='score', hue='lm',
                     markers={m: 'o' for m in df.lm.unique()}, style='lm', ci='sd', dashes=False,
                     palette=colors[2:2 + len(df.lm.unique())])

    _change_legend(g.legend(), {
        'bert-base-uncased': 'bert-base',
        'bert-large-uncased': 'bert-large',
        'albert-xxlarge-v2': 'albert-xxlarge',
    },
                   title='')
    plt.xlabel('# Examples')
    plt.ylabel('Normalized Score')
    plt.title(f'TextWorld Commonsense')
    plt.show()
    _save(fig, 'twc')


def main():
    fname = 'results.txt'
    if not fname in os.listdir():
        print('No results.txt file available. Run the experiment at least once.')
        return
    df = load_results(fname)
    plot(df)


if __name__ == '__main__':
    main()