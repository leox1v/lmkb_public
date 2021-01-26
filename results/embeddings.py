from absl import app, flags, logging
import os
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns
from collections import defaultdict

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', 'out/res-emb',
        'Input directory.')
flags.DEFINE_string('output_dir', 'out/res-emb/plots',
        'Output directory.')
flags.DEFINE_enum('dim_reduction', 'tsne', ['pca', 'tsne'],
        'The dimension reduction technique to use.')


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

def save(fig, fname: str):
    fname = os.path.join(FLAGS.output_dir, f'{fname}.pdf')
    os.makedirs('/'.join(fname.split('/')[:-1]), exist_ok=True)
    logging.info(f'Saving plot {fname}.')
    fig.savefig(fname,
            format='pdf',
            bbox_inches='tight')

def plt_style():
    sns.set(font_scale=1)
    plt.style.use(['seaborn-white'])

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot(individual_plots=True, n_per_relation=400):
    # Load data
    fnames = list(filter(lambda x: x.endswith('.csv'), get_all_files(FLAGS.input_dir)))
    # Hack to make the order right.
    fnames = [fnames[i] for i in [0, 2, 1, 3]]

    relation_strs = {k: v['template'].replace('[X]', '[s]').replace('[Y]', '[o]')
            for k, v in get_trex_relations().items()}
    plt_style()

    relations = sorted([
            'P19', # [X] was born in [Y] .
            'P20', # [X] died in [Y] .
            'P279', # [X] is a subclass of [Y] .
            # 'P37', # The official language of [X] is [Y] .
            # 'P413', # [X] plays in [Y] position .
            # 'P449', # [X] was originally aired on [Y] .
            'P47', # [X] shares border with [Y] .
            # 'P138', # [X] is named after [Y] .
            # 'P364', # The original language of [X] is [Y] .
            'P463', # [X] is a member of [Y] .
            # 'P101', # [X] works in the field of [Y] .
            # 'P106', # [X] is a [Y] by profession .
            # 'P527', # [X] consists of [Y] .
            # 'P102', # [X] is a member of the [Y] political party .
            # 'P530', # [X] maintains diplomatic relations with [Y] .
            # 'P176', # [X] is produced by [Y] .
            'P27', # [X] is [Y] citizen .
            # 'P407', # [X] was written in [Y] .
            'P30', # [X] is located in [Y] .
            # 'P178', # [X] is developed by [Y] .
            # 'P1376', # [X] is the capital of [Y] .
            # 'P131', # [X] is located in [Y] .
            # 'P1412', # [X] used to communicate in [Y] .
            # 'P108', # [X] works for [Y] .
            # 'P136', # [X] plays [Y] music .
            # 'P17', # [X] is located in [Y] .
            # 'P39', # [X] has the position of [Y] .
            # 'P264', # [X] is represented by music label [Y] .
            # 'P276', # [X] is located in [Y] .
            # 'P937', # [X] used to work in [Y] .
            # 'P140', # [X] is affiliated with the [Y] religion .
            # 'P1303', # [X] plays [Y] .
            # 'P127', # [X] is owned by [Y] .
            # 'P103', # The native language of [X] is [Y] .
            # 'P190', # [X] and [Y] are twin cities .
            # 'P1001', # [X] is a legal term in [Y] .
            'P31', # [X] is a [Y] .
            # 'P495', # [X] was created in [Y] .
            # 'P159', # The headquarter of [X] is in [Y] .
            # 'P36', # The capital of [X] is [Y] .
            # 'P740', # [X] was founded in [Y] .
            # 'P361', # [X] is part of [Y] .
    ])
    k = len(relations)
    # color_by_index = lambda x: plt.cm.Dark2(x/float(k))
    color_by_index = lambda x: plt.cm.tab20(x/float(k))

    if individual_plots:
        fig = plt.figure()
    else:
        fig, axes = plt.subplots(1, len(fnames), figsize=(25, 5))

    for i, fname in enumerate(fnames):
        # Set axis.
        if not individual_plots:
            plt.sca(axes[i])

        # plt.gca().set_aspect(1.)

        df = pd.read_csv(fname)
        df = df[df.apply(lambda row: row.rel in relations, axis=1)]
        df = df.groupby('rel').head(n_per_relation)

        # Fill up relations
        for rel in relations:
            delta = n_per_relation - len(df[df.rel == rel])
            if delta > 0:
                df = df.append(df[df.rel == rel].sample(n=delta, replace=True))

        assert len(df) == n_per_relation * len(relations), \
                'Not all relations have enough samples.\n' + \
                '\n'.join([f'{rel}\t{len(df[df.rel == rel])}' for rel in relations])
        df.loc[:, 'cls_embedding'] = df['cls_embedding'].apply(literal_eval)

        embeddings_per_relation = df.groupby('rel')['cls_embedding'].apply(list)
        embeddings = np.array([emb for r in relations for emb in embeddings_per_relation[r]])
        colors = [color_by_index(i) for i, r in enumerate(relations) for _ in range(len(embeddings_per_relation[r]))]
        average_embeddings = embeddings_per_relation.apply(lambda x: np.mean(x, axis=0))
        average_embeddings = np.array([average_embeddings[r] for r in relations])
        all_embeddings = np.concatenate([embeddings, average_embeddings])
        if FLAGS.dim_reduction == 'pca':
            embeddings_proj = PCA().fit_transform(all_embeddings)[:, :2]
        elif FLAGS.dim_reduction == 'tsne':
            embeddings_proj = TSNE(n_components=2).fit_transform(all_embeddings)

        x, y = zip(*embeddings_proj)

        plt.scatter(x[:len(embeddings)], y[:len(embeddings)], c=colors,
                s=2.)

        # Plot ellipses.
        for i, r in enumerate(relations):
            xx, yy = zip(*embeddings_proj[i*n_per_relation:(i+1)*n_per_relation, :])
            confidence_ellipse(np.array(xx), np.array(yy), plt.gca(),
                    edgecolor=color_by_index(i), n_std=2, linestyle='solid')


        # Plot the means.
        for i, (xx, yy, relation) in enumerate(zip(x[len(embeddings):], y[len(embeddings):], relations)):
            mean_plt = plt.scatter(xx, yy,
                color=color_by_index(i),
                marker='x',
                s=50.,
                label=relation_strs[relation])


        # Annotate with the relations.
        # for i, r in enumerate(relations):
        #     rel_str = relation_strs[r]
        #     xx, yy = embeddings_proj[len(embeddings) + i]
        #     plt.gca().annotate(rel_str, (xx, yy),
        #             fontsize=5., ha='center')

        plt.locator_params(nbins=1)

        path = fname.replace(FLAGS.input_dir, '').strip('/')
        data = path.split('/')[0]
        model = path.split('/')[1]
        infos = {x.split('-')[0]: x.split('-')[1]
                for x in path.split('/')[2].strip('.csv').split('_')}

        example_type_title = {'nl': 'NL Template',
                'par': '({}; {}) Template'}.get(infos['nl'], '')
        title = ' w/ '.join([example_type_title, infos['pe'] + ' Examples'])
        plt.gca().set_title(title, fontsize=20)

        if individual_plots:
            plt.gca().legend(fontsize=8.)
            save(fig, f'embeddingplot_pe_{infos["pe"]}_pt_{infos["nl"]}')

    plt.gca().legend(loc='upper center',
            bbox_to_anchor=(-1.3125, -0.1), fontsize=20., ncol=int(np.ceil(k/2)),
            fancybox=True,
            shadow=True)
    if not individual_plots:
        # handles, labels = axes[-1].get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center', fontsize=8.)
        sns.despine()
        save(fig, 'embeddingplot_overview')


def main(argv):
    # plot(individual_plots=True,
    #         n_per_relation=30)
    plot(individual_plots=False,
            n_per_relation=500)

if __name__ == "__main__":
    app.run(main)
