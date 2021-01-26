import torch
import os
import pandas as pd

from absl import app, logging, flags
from collections import defaultdict
from tqdm import tqdm
from time import time
from lm.lm import LM

from data.data import PrimingDataLoader
from data.merge_data import process_df_per_type

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', 'out/res',
        'Output directory.')
flags.DEFINE_string('model', 'bert-base-cased',
        'Model name for 🤗/transformers model.')
flags.DEFINE_enum('data', 'TREx', ['TREx', 'ConceptNet', 'Google_RE', 'BATS'],
        'Dataset.')
flags.DEFINE_integer('num_predictions', 20,
        'Number predictions per example.')
flags.DEFINE_integer('num_priming_examples', 0,
        'Number of priming examples for each prediction.')
flags.DEFINE_integer('batch_size', 32,
        'Batch size per gpu.')
flags.DEFINE_enum('priming_type', 'nl', ['nl', '{} -> {}', '{} => {}', '({}; {})'],
        'How to provide the priming examples. To use natural language pick "nl".')
flags.DEFINE_bool('use_close_examples', False,
        'Whether or not to use subjects for priming that are close in latent space.')
flags.DEFINE_bool('save_results', True,
        'Whether or not to save the results to the output directory.')
flags.DEFINE_bool('use_common_vocab', True,
        'Whether or not to use shared vocab between the models.')
flags.DEFINE_integer('random_seed', 0,
        'The used random seed.')
flags.DEFINE_bool('save_cls_embedding', False,
        'Save the cls embedding of the predictions.')


def main(argv):
    use_common_vocab = FLAGS.use_common_vocab and ('-cased' in FLAGS.model or 'albert' in FLAGS.model)
    lm = LM(FLAGS.model,
            use_common_vocab=use_common_vocab)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_devices = torch.cuda.device_count() if 'cuda' in device else 1
    if num_devices > 1:
        lm.model = torch.nn.DataParallel(lm.model)
    lm.to(device)

    dl = PrimingDataLoader(batch_size=FLAGS.batch_size * num_devices,
                           identifier=FLAGS.data,
                           num_priming_examples=FLAGS.num_priming_examples,
                           priming_type=FLAGS.priming_type,
                           use_close_examples=FLAGS.use_close_examples,
                           random_seed=FLAGS.random_seed,
                           vocab='vocab/common_vocab_cased.txt' \
                                   if use_common_vocab \
                                   else lm.vocab,
    )

    data = defaultdict(list)

    start_time = time()
    for idx, batch in enumerate(tqdm(dl,
        desc=f'Evaluate "{FLAGS.model}" on {FLAGS.data}',
        total=len(dl))):

        x = lm.prepare_input(batch['x_primed'])
        out = lm(x, k=FLAGS.num_predictions,
                save_cls_embedding=FLAGS.save_cls_embedding)
        out = {**batch, **out}

        for k, v in out.items():
            if v:
                data[k] += v

    runtime = time() - start_time
    data = pd.DataFrame(data)

    summary = process_df_per_type(
            df=data,
            data=FLAGS.data,
            model=FLAGS.model,
            num_examples=FLAGS.num_priming_examples,
            priming_type=FLAGS.priming_type,
            use_close_examples=FLAGS.use_close_examples,
            random_seed=FLAGS.random_seed)
    p1 = summary[summary.data == FLAGS.data]['p@1'][0]
    print(summary)

    priming_type = {
            '{} -> {}': 'rarr',
            '{} => {}': 'Rarr',
            '({}; {})': 'par',
    }.get(FLAGS.priming_type, FLAGS.priming_type)

    fname = f'{FLAGS.data}/{FLAGS.model}/'\
            f'pe-{FLAGS.num_priming_examples}_'\
            f'nl-{priming_type}_'\
            f'ce-{FLAGS.use_close_examples}_'\
            f'rs-{FLAGS.random_seed}.csv'
    fname = os.path.join(FLAGS.output_dir, fname)
    os.makedirs('/'.join(fname.split('/')[:-1]), exist_ok=True)
    logging.info(f'Achieved total P@1 of {p1*100:.2f}.')
    logging.info(f'Total runtime was {runtime:.1f} seconds.')
    average_input_token_length = data.apply(lambda row: len(row.x_primed.split()),
            axis=1).mean()
    logging.info(f'Average input token length: {average_input_token_length:.1f}. '\
            f'Number of total datapoints: {len(data)}.')
    if FLAGS.save_results:
        logging.info(f'Saving {fname}.')
        data = data.to_csv(fname)
    else:
        logging.warning('Result not saved. Set "--save_results" to save.')

if __name__ == '__main__':
    app.run(main)
