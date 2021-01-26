import pandas as pd
import os
from absl import app, logging, flags
from tqdm import tqdm
from tables import generate_results_table
import seaborn as sns
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS

flags.DEFINE_string('summary_file', 'out/summary/df.csv',
        'Dataframe file from which to load the results summaries.')
flags.DEFINE_string('output_dir', 'out/summary',
        'Output directory to save the summaries.')
flags.DEFINE_bool('precision_plots', True,
        'Whether or not to generate the precision plots.')
flags.DEFINE_bool('results_table', True,
        'Whether or not to generate the results table.')

class Plotter:
    def __init__(self, df: pd.DataFrame, plot_dir: str):
        self.df = df
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        self.palette = sns.color_palette()

        def convert_nl_to_priming_type(priming_type, use_nl):
            if priming_type == priming_type:
                # if it's not NaN everything is fine.
                return priming_type
            return 'nl' if use_nl else 'rarr'

        if 'use_natural_language' in self.df.columns:
            self.df.priming_type = self.df.apply(lambda row:
                    convert_nl_to_priming_type(row.priming_type,
                        row.use_natural_language), axis=1)

    def _plt_style(self):
        sns.set(font_scale=1.8)
        plt.style.use(['seaborn-whitegrid'])

    def _save(self, fig, fname: str):
        fname = os.path.join(self.plot_dir, f'{fname}.pdf')
        logging.info(f'Saving plot {fname}.')
        fig.savefig(fname,
                format='pdf',
                bbox_inches='tight')

    def _change_legend(self, legend, mapping, title):
        for t in legend.texts:
            if t.get_text() in mapping:
                t.set_text(mapping[t.get_text()])
        legend.set_title(title)

    def bats_plots(self):
        self._plt_style()

        models = ['bert-large-uncased', 'bert-base-uncased', 'albert-xxlarge-v2']
        use_close_examples = False
        priming_type = 'par'
        adjust = False
        data = 'BATS'

        df = self.df[(self.df.priming_type == priming_type) \
                & (self.df.use_close_examples == use_close_examples) \
                & (self.df.data == data)]
        if adjust:
            df.loc[:, 'p@1'] = df['p@1'] * 1522/2000

        fig = plt.figure()

        # Add baseline
        for name, val in [('GloVe', 0.285), ('SVD', 0.221)]:
            plt.plot([0, 20], [val, val], '--', color='k', linewidth=1)
            plt.text(-0.95, val+0.003, name, fontsize=10)


        g = sns.lineplot(data=df, x='num_examples', y='p@1',
                hue='model', hue_order=models,
                style='model',
                markers={k: 'o' for k in models},
                dashes=False)

        priming_str = {
            'Rarr': '[s] => [o]',
            'rarr': '[s] -> [o]',
            'par': '([s]; [o])',
        }[priming_type]

        self._change_legend(g.legend(), {
            'bert-base-uncased': 'bert-base',
            'bert-large-uncased': 'bert-large',
            'albert-xxlarge-v2': 'albert-xxlarge',
        },
        title='')

        plt.xlabel('# Examples')
        plt.title(f'BATS {priming_str}')
        plt.ylim(bottom=0)

        fname = f'BATS_plot_{priming_type}'

        self._save(fig, fname)

        # Plot across categories
        model = 'albert-xxlarge-v2'
        use_close_examples = False
        priming_type = 'par'
        num_examples = 10

        df = self.df[(self.df.priming_type == priming_type) \
                & (self.df.use_close_examples == use_close_examples) \
                & (self.df.model == model) \
                & (self.df.num_examples == num_examples) \
                & self.df.apply(lambda row: len(row.data.split('_')) > 1,
                    axis=1)]
        df.loc[:, 'data'] = df.apply(lambda row: row.data.split('_')[-1],
                axis=1)
        fig = plt.figure(figsize=(20, 5))
        with open('data/bats_benchmark_glove.txt', 'r') as f:
            benchmark = [(l.split()[0], float(l.split()[1])/667)
                    for l in f.readlines()]
        dfb = pd.DataFrame(benchmark, columns=['data', 'p@1'])
        dfb.loc[:, 'model'] = 'GloVe'
        df = pd.concat([df, dfb])
        order = ['I', 'D', 'L', 'E']

        # Specify data categories
        bats_dir = 'data/BATS_3.0'
        fnames = [f.split('.')[0] for d in os.listdir(bats_dir)
                if not '.' in d for f in os.listdir(os.path.join(bats_dir, d))]
        fnames = {f.split()[0]: f for f in fnames}
        df.data = df.apply(lambda row: fnames[row.data], axis=1)

        # Specify "solvable"
        with open('data/BATS_3.0/solvable.txt', 'r') as f:
            solvable = {fnames[l.split('\t')[0]]: float(l.split('\t')[1].strip('\n'))/50 for l in f.readlines()}
        df2 = df.copy()
        df2['p@1'] = df2.apply(lambda row: solvable[row.data] if row.model == 'albert-xxlarge-v2' else 0., axis=1)
        df2['model'] = df2.apply(lambda row: f'_{row.model}', axis=1)

        g2 = sns.barplot(x='data', y='p@1', data=df2, hue='model',
                linewidth=0.25, facecolor=(1, 1, 1, 0), edgecolor=".2", ci=None,
                order=sorted(df2.data.unique(),
                    key=lambda x: (order.index(x[0]), x)))

        g = sns.barplot(x='data', y='p@1', data=df,
                hue='model',
                order=sorted(df.data.unique(),
                    key=lambda x: (order.index(x[0]), x)))

        plt.xticks(rotation=90)
        plt.xlabel('Category')
        plt.ylabel('P@1')
        plt.title(f'BATS Albert (10 examples)')
        self._change_legend(g.legend(), {
            'albert-xxlarge-v2': 'albert-xxlarge',
        },
        title='')

        fname = 'BATS_category_plot'
        self._save(fig, fname)



    def arrow_plots(self):
        self._plt_style()

        model = 'bert-large-cased'
        data = 'TREx'
        priming_types = ['nl', 'par', 'rarr', 'Rarr']

        df = self.df[(self.df.model == model) & \
                (self.df.data == data) & \
                self.df.apply(lambda row: row.priming_type in priming_types,
                    axis=1)]
        fig = plt.figure()
        g = sns.lineplot(data=df, x='num_examples', y='p@1',
                hue='priming_type', hue_order=priming_types,
                style='priming_type',
                markers={k: 'o' for k in priming_types},
                dashes=False)
        # Add baseline
        y_start = df[(df.priming_type == 'nl') & (df.num_examples == 0)]['p@1'].iloc[0]
        plt.plot([0, 20], [y_start, y_start], '--', color=self.palette[priming_types.index('nl')])

        self._change_legend(g.legend(), {
            'nl': 'NL Template',
            'Rarr': '[s] => [o]',
            'rarr': '[s] -> [o]',
            'par': '([s]; [o])',
        },
        title='')

        plt.xlabel('# Examples')
        plt.ylabel('P@1')
        plt.title(f'{data} BERT-large')
        plt.ylim(bottom=0)

        fname = 'arrow_plot'

        self._save(fig, fname)

    def precision_plots(self):
        self._plt_style()

        # models_all = ['bert-large-cased', 'bert-base-cased', 'albert-xxlarge-v2',
        #         't5-large', 'bert-large-uncased', 'bert-base-uncased']
        models_all = self.df.model.unique()
        for data in self.df.data.unique(): #['TREx', 'ConceptNet', 'Google_RE']:
            for priming_type in self.df.priming_type.unique():
                for use_ce in self.df.use_close_examples.unique():
                    for y in ['p@1']: #, 'mrr', 'target_likelihood']:
                        fig = plt.figure()
                        x = 'num_examples'
                        df = self.df[(self.df.priming_type == priming_type) & \
                                (self.df.use_close_examples == use_ce) & \
                                (self.df.data == data) & \
                                self.df.apply(lambda row: row.model in models_all, axis=1)]
                        models = [m for m in models_all if m in df.model.unique()]

                        if any([m.startswith('t5') for m in models]):
                            # Adjustment (token-target mismatch)
                            def p1_adjustment_t5(row):
                                val = row['p@1']
                                if row.model.startswith('t5'):
                                    if row.data == 'TREx':
                                        return val * 0.7789
                                    elif row.data == 'ConceptNet':
                                        return val * 0.379
                                return val
                            df['p@1'] = df.apply(p1_adjustment_t5, axis=1)

                        g = sns.lineplot(data=df, x=x, y=y, hue='model', style='model',
                                        hue_order=models,
                                        markers={k: 'o' for k in models},
                                        dashes=False)
                        added_plots = 0
                        for i, model in enumerate(models):
                            try:
                                y_start = df[(df.model == model) & (df.num_examples == 0)][y].iloc[0]
                            except:
                                continue

                            x_min = df[(df.model == model)][x].min()
                            x_max = df[x].max()
                            plt.plot([x_min, x_max], [y_start, y_start], '--', color=self.palette[i])
                            added_plots += 1

                        if added_plots == 0:
                            continue

                        self._change_legend(g.legend(), {
                            'bert-base-cased': 'bert-base',
                            'bert-large-cased': 'bert-large',
                            'bert-base-uncased': 'bert-base',
                            'bert-large-uncased': 'bert-large',
                            'albert-xxlarge-v2': 'albert-xxlarge',
                        },
                        title='')
                        plt.ylabel({
                            'p@1': 'P@1',
                            'mrr': 'MRR',
                            'target_likelihood': 'Avg. Probability',
                        }[y])
                        plt.xlabel('# Examples')
                        plt.title(f'{data}')
                        plt.ylim(bottom=0)
                        fname = f'{y}_data-{data}_pt-{priming_type}_ce-{use_ce}'
                        self._save(fig, fname)
                        plt.close('all')

def main(argv):
    df = pd.read_csv(FLAGS.summary_file)

    if FLAGS.results_table:
        generate_results_table(df,
                output_dir=FLAGS.output_dir)
    plotter = Plotter(df,
            plot_dir=os.path.join(FLAGS.output_dir, 'plots/'))
    if FLAGS.precision_plots:
        plotter.precision_plots()

    if 'df_arr' in FLAGS.summary_file:
        plotter.arrow_plots()

    if 'df_bats' in FLAGS.summary_file:
        plotter.bats_plots()



if __name__ == "__main__":
    app.run(main)

