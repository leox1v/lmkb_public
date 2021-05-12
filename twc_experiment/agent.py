import numpy as np
import pandas as pd
import json
import editdistance
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pprint import pformat
from textworld import EnvInfos
from typing import Any, Dict, List, Union
from torch import nn
from lm import LM, AdaptiveObjLocPriming
from utils import admissible_commands_filter, Observation, HistoryEntry, get_inventory_list, get_all_entities_of_types, location_from_description


@dataclass
class AgentConfig:
    name: str
    model: Union[nn.Module, LM]
    extras: dict = field(default_factory=dict)


def get_agent(config: AgentConfig):
    if config.name == 'random':
        return RandomAgent(config)
    elif config.name == 'lmkb':
        return LMPriorAgent(config, random=False)
    elif config.name == 'lmkb-random':
        return LMPriorAgent(config, random=True)
    else:
        raise NotImplementedError(f'Agent {config.name} is not implemented.')


class BaseAgent(ABC):

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.config.extras = {**self.config.extras, **dict(type=type(self))}

        self.history = None
        self.is_done = False

        self.episode_count = 0

    @abstractmethod
    def act(self, obs: str, score: int, done: bool, infos: Dict[str, Any]
            ) -> str:
        pass

    def new_episode(self, observation: str, infos: Dict[str, Any]):
        """Start a new episode."""
        self.history = []
        self.is_done = False
        self.episode_count += 1
        self.score = 0

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(
                description=True,
                inventory=True,
                admissible_commands=True,
                won=True,
                lost=True,
                location=True,
                last_action=True,
                game=True,
                facts=True,
                entities=True,
                max_score=True,
        )

    def __str__(self):
        return pformat(asdict(self.config))


PLACEHOLDER_CMD = 'look'


class LMPriorAgent(BaseAgent):

    def __init__(self, config: AgentConfig, random: bool = False):
        super().__init__(config)
        self.model = config.model
        self.random = random
        self.sampling = config.extras.get('sampling', True)
        self.template = {'arrow': '{} -> {}',
                         'semicolon': '{}; {}',
                         'nl': '{} belongs to {}'}[config.extras.get('template_id', 'arrow')]
        self.priming_examples = config.extras.get('num_priming_examples', 10)

        with open('games/twc_objects.json') as json_file:
            self.all_objects = {k: v for k, v in json.load(json_file).items()
                                if v['type'] == 'o' or v['type'] == 'f'}

    def act(self, obs: str, score: int, done: bool, infos: Dict[str, Any]
            ) -> str:
        if self.is_done and done:
            # The 'done' appeared the second time. Episode is over and we're
            # just waiting for the other agents in the batch to finish. Just return
            # a placeholder command and an empty tranisition.
            return PLACEHOLDER_CMD

        observation = Observation(obs=obs,
                      description=infos['description'],
                      inventory=get_inventory_list(infos['inventory']))
        admissible_commands = admissible_commands_filter(infos['admissible_commands'])
        history_entry = HistoryEntry(observation=observation,
                                     admissible_commands=admissible_commands,
                                     chosen_command_idx=-1,
                                     score=score)
        self.history.append(history_entry)

        self.update_prior()

        if done:
            self.is_done = True
            return PLACEHOLDER_CMD

        try:
            action = self.get_action(observation, admissible_commands)
        except:
            action = np.random.choice(admissible_commands)

        # Overwrite the placeholder action in the history.
        self.history[-1].chosen_command_idx = admissible_commands.index(action)
        self.score = score

        return action

    def get_action(self, observation: str, admissible_commands: List[str]) -> str:
        """Decide for an action."""
        if not observation.inventory:
            successfully_stored = [obj for obj, stored in zip(self.prior.index, (self.prior == 1).any(axis='columns')) if stored]
            # Randomly choose one of the 'take' commands.
            return np.random.choice([a for a in admissible_commands
                if a.startswith('take') and not any([obj in a.split('from')[0] for obj in successfully_stored])])

        obj = sorted(self.objects, key=lambda x: editdistance.eval(x, observation.inventory[0]))[0]
        probability = self.prior.loc[obj]
        if not self.sampling:
            probability = (np.arange(len(probability)) == probability.astype(float).argmax()).astype(float)
        container = np.random.choice(self.prior.columns, p=probability/sum(probability))
        self.prior.loc[obj, container] = 'T'

        cmd = list(filter(lambda cmd: container in cmd, admissible_commands))[0]

        return cmd

    def update_prior(self):
        if not (self.prior == 'T').any(axis=None):
            # Nothing to update
            return
        # Find the tried combination, i.e. the cell with 'T'.
        idxs = [(obj, container) for obj, container in zip(*np.where(self.prior.to_numpy() == 'T'))]
        assert len(idxs) <= 1, 'More than one cell was set to T=tried.'
        # cheat
        try:
            success = self.prior.columns[idxs[0][1]] in self.all_objects[self.prior.index[idxs[0][0]]]['locations']
        except:
            print(f'Seems like {self.prior.index[idxs[0][0]]} is not in all_objects.')
            success = True
        if success:
            # Set all the other values in the row to 0.
            self.prior.iloc[idxs[0][0]] = 0.
            self.prior.iloc[idxs[0]] = 1.
        else:
            self.prior.iloc[idxs[0]] = 0.
            # Re-normalize.
            row_values = self.prior.iloc[idxs[0][0]].to_numpy()
            row_sum = row_values.sum()
            self.prior.iloc[idxs[0][0]] = [v/row_sum for v in row_values]


    def new_episode(self, observation, infos):
        """Start a new episode."""
        super().new_episode(observation, infos)

        self.location = location_from_description(infos['description'])
        self.containers = get_all_entities_of_types(infos['facts'], ['c', 's'])
        self.objects = get_all_entities_of_types(infos['facts'], ['o', 'f'])

        if self.random:
            self.prior = {obj: [1./len(self.containers)]*len(self.containers) for obj in self.objects}
        else:
            self.priming = AdaptiveObjLocPriming(self.objects, self.containers, n=self.priming_examples,
                                                 template=self.template)
            self.prior = {obj: np.array(self.model.continuation_prob(
                self.priming.generate_input(obj), self.containers)['probs']) + 0.0001
                     for obj in self.objects}
        self.prior = pd.DataFrame.from_dict(self.prior, orient='index',
                                            columns=self.containers)

class RandomAgent(BaseAgent):

    def __init__(self, config: AgentConfig):
        super().__init__(config)

    def act(self, obs: str, score: int, done: bool, infos: Dict[str, Any]
            ) -> str:
        return np.random.choice(infos['admissible_commands'])