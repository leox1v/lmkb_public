from absl import app, flags
from collections import defaultdict
import numpy as np
from torch import cuda
from tqdm import tqdm
from typing import Optional

from agent import get_agent, AgentConfig
from utils import location_from_description, parse_single_batch_output
from games import dataset
from lm import LM

flags.DEFINE_string('game_path', 'games/hard/test',
                    'Directory from which to load the games.')
flags.DEFINE_enum('agent', 'lmkb',
                  ['random', 'lmkb', 'lmkb-random'],
                  'Type of agent.')
flags.DEFINE_enum('lm', 'albert-xxlarge-v2', ['albert-xxlarge-v2', 'bert-base-uncased', 'bert-large-uncased'],
        'The language model to use for lmkb.')
flags.DEFINE_enum('priming_template', 'arrow', ['arrow', 'nl', 'semicolon'],
                  'The template used for the priming.')
flags.DEFINE_integer('num_priming_examples', 10,
                  'The number of priming examples used.')
flags.DEFINE_integer('max_num_steps_per_game', 50,
                     'Maximum number of steps per game.')
flags.DEFINE_integer('num_episodes', 15,
                     'Number of episodes.')
flags.DEFINE_boolean('verbose', False, 'Verbosity.')
flags.DEFINE_boolean('use_sampling', True, 'Use sampling or greedy choice.')

FLAGS = flags.FLAGS

def get_model(agent_name: str, lm_identifier: Optional[str] = None):
    if agent_name == 'lmkb':
        model = LM(lm_identifier)
    else:
        model = None
    if model and cuda.is_available():
        model.to('cuda:0')
    return model


def main(argv):
    model = get_model(FLAGS.agent, FLAGS.lm)
    agent_config = AgentConfig(
            model=model,
            name=FLAGS.agent,
            extras=dict(
                sampling=FLAGS.use_sampling,
                template_id=FLAGS.priming_template,
                num_priming_examples=FLAGS.num_priming_examples,
            )
    )

    agent = get_agent(agent_config)

    env, _ = dataset.get_game_env(
            game_path=FLAGS.game_path,
            requested_infos=agent.infos_to_request,
    )

    episode_scores = defaultdict(list)
    num_steps = []

    for _ in tqdm(range(FLAGS.num_episodes)):
        observation, info = parse_single_batch_output(*env.reset())
        location = location_from_description(info['description'])
        agent.new_episode(observation, info)
        score, done = None, None

        step = 0
        while not done and step < FLAGS.max_num_steps_per_game:
            command = agent.act(observation, score, done, info)
            if FLAGS.verbose:
                print(observation)
                print(command)
            observation, score, done, info = parse_single_batch_output(*env.step([command]))
            step += 1

        # Let the agents know the game is over.
        agent.act(observation, score, done, info)

        episode_scores[location].append(score/info['max_score'])
        num_steps.append(step)
        print(f'Episode score: {episode_scores[location][-1]:.2f}. Steps: {step}. Location: {location}')

    # Save the stats.
    print()
    flat_episode_scores = [v for v_list in episode_scores.values() for v in v_list]
    score = np.mean(flat_episode_scores)
    steps = np.mean(num_steps)
    print(f'Average episode score: {score}')
    print('Average episode score: {}'.format('. '.join([f'{loc}: {np.mean(v):.2f}' for loc, v in episode_scores.items()])))
    print(f'Average number of steps: {steps}')

    def _info_str():
        return f'games:{FLAGS.game_path},agent:{FLAGS.agent},lm:{FLAGS.lm},' + \
        f'priming_template:{FLAGS.priming_template},num_priming_examples:{FLAGS.num_priming_examples}'
    with open('results.txt', 'a+') as f:
        f.write(_info_str() + f',score:{score},steps:{steps}' + '\n')


if __name__ == "__main__":
    app.run(main)
