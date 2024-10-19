import itertools
from copy import deepcopy
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


# Declare spaces
class State(Enum):
    b = auto()
    c = auto()
    d = auto()


class Action(Enum):
    x = auto()
    y = auto()


class Trajectory:
    states: list[State]
    actions: list[Action]
    rewards: NDArray[np.int64]

    def __init__(self, states, actions, rewards) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards


trajectories = [
    Trajectory(
        [State.d, State.b, State.d, State.c],
        [Action.x, Action.x, Action.y],
        np.array([1.5, -1, 2]),
    ),
    Trajectory(
        [State.b, State.d, State.b, State.b, State.d, State.d, State.c],
        [Action.x, Action.y, Action.y, Action.x, Action.x, Action.x],
        np.array([-1, 2, 0, -1, 1.5, 1.5]),
    ),
    Trajectory(
        [State.b, State.d, State.d, State.d, State.b, State.c],
        [Action.y, Action.x, Action.x, Action.y, Action.y],
        np.array([0, 1.5, 1.5, 2, 0]),
    ),
]

# Initialize vhat
nonterminal_states = [State.b, State.d]


def initilize_for_each_state_action_pair(val):
    return {
        (state, action): deepcopy(val)
        for state, action in itertools.product(nonterminal_states, Action)
    }


qhat = initilize_for_each_state_action_pair(0.0)

# Initialize set of G
returns = initilize_for_each_state_action_pair([])

# Initialize N(s)
num_visits = initilize_for_each_state_action_pair(0)

for traj in trajectories:
    # For each state-action pair... (excluding state c since we end at that state)
    for state, action in itertools.product(nonterminal_states, Action):
        pair = (state, action)

        # If the pair in question isn't in the trajectory at all, then don't
        # attempt to update qhat
        if pair not in zip(traj.states[:-1], traj.actions):
            continue

        # For each step in the trajectory...
        for t, (s, a, r) in enumerate(
            zip(traj.states[:-1], traj.actions, traj.rewards)
        ):
            # Skip if the step does not match the state-action pair in question
            if s != state or a != action:
                continue

            # We aren't told about any discount, so the return is just the  sum
            # of the rewards from the current time forward
            ret = float(np.sum(traj.rewards[t:]))
            returns[pair].append(ret)
            num_visits[pair] += 1

        qhat[pair] = float((1 / num_visits[pair]) * np.sum(returns[pair]))

for k, v in qhat.items():
    print(f"({k[0].name}, {k[1].name}): {v}")
