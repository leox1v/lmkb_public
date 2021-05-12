from dataclasses import dataclass
from typing import Optional, List
import numpy as np


class Observation(object):
    def __init__(self, obs: str, description: str, inventory: str):
        # Clean the obs by removing the description and the welcome text.
        self.obs = obs
        if description in self.obs:
            self.obs = self.obs.replace(description, '')
        if 'Welcome to TextWorld!' in self.obs:
            self.obs = self.obs.split('Welcome to TextWorld! ')[1].strip()

        self.description = description
        self.inventory = inventory

    def __str__(self):
        return f'{self.obs}\n{self.description}\n{self.inventory}'


@dataclass
class HistoryEntry:
    observation: Observation
    admissible_commands: List[str]
    chosen_command_idx: int
    score: int


def parse_single_batch_output(*args):
    res = []
    for arg in args:
        if isinstance(arg, list) or isinstance(arg, tuple):
            parsed_arg = arg[0]
        elif isinstance(arg, dict) and all([isinstance(a, list) or isinstance(a, tuple) for a in arg.values()]):
            parsed_arg = {k: v[0] for k, v in arg.items()}
        else:
            raise NotImplementedError()
        res.append(parsed_arg)
    return res


def admissible_commands_filter(admissible_commands: List[str],
                               k: Optional[int]=None) -> List[str]:
    """Filters out useless commands.

    Args:
        admissible_commands: The list of admissible_commands.
        k: Maximum number of final commands.

    Returns:
        Filtered list of admissible_commands.
    """
    # Filter out all 'examine' commands except for the cookbook.
    admissible_commands = filter(
            lambda cmd: (not 'examine' in cmd or 'cookbook' in cmd), admissible_commands
    )

    # Filter out specified commands.
    individual_commands_to_remove = ['look', 'inventory']
    admissible_commands = list(filter(
            lambda cmd: cmd not in individual_commands_to_remove, admissible_commands
    ))

    # Remove other type of commands.
    commands_to_remove = ['close', 'drop', 'open']
    admissible_commands = list(filter(
        lambda cmd: all([not cmd.startswith(forbidden_cmd) for forbidden_cmd in commands_to_remove]),
        admissible_commands
    ))

    if k is not None and len(admissible_commands) > k:
        admissible_commands = np.random.choice(list(admissible_commands),
                                               size=k,
                                               replace=False)

    return list(admissible_commands)

def get_inventory_list(inventory_str: str) -> List[str]:
    """Turns the textual description of the inventory to a list of inventory items."""
    if inventory_str == 'You are carrying nothing.':
        return []
    inventory = inventory_str.replace('You are carrying:', '')
    inventory = [item.strip() for item in inventory.split('\n ') if item.strip()]

    # clean away articles etc.
    to_remove = ['a', 'an', 'some', 'pair', 'of']
    for i, obj in enumerate(inventory):
        split_obj = obj.split()
        inventory[i] = ' '.join([o for o in split_obj if o not in to_remove])

    return inventory

def get_all_entities_of_types(facts: List, t: List[str]) -> List[str]:
    entities = set()
    for fact in facts:
        entities |= set([e.name for e in fact.arguments if e.type in t])
    return list(entities)

def location_from_description(description: str) -> str:
    return description.split('\n')[0].strip('=- ').lower()


