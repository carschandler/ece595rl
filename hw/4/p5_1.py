import itertools

import numpy as np
import xarray as xr
from p1_1 import action_space, gamma, reward_fn, state_space, trans_fn
from p2_2_4 import norm_state_occupancy

policy = xr.DataArray(
    np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    ),
    coords=dict(state=state_space, action=action_space),
)


policy_prime = xr.DataArray(
    np.array(
        [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    ),
    coords=dict(state=state_space, action=action_space),
)

q_prime = xr.DataArray(
    np.array(
        [
            [13.0, 12.0],
            [14.0, 13.5],
            [10.0, 15.0],
        ]
    ),
    coords=dict(state=state_space, action=action_space),
)

expected_trans_fn_prime = np.reshape(
    [
        np.dot(
            trans_fn.sel(current_state=s, next_state=sn),
            policy_prime.sel(state=s),
        )
        for s, sn in itertools.product(state_space, state_space)
    ],
    (3, 3),
)

expected_reward_fn_prime = (reward_fn * policy_prime).sum(dim="action")

value_prime = (
    np.linalg.inv(
        np.identity(len(state_space)) - gamma * expected_trans_fn_prime
    )
    @ expected_reward_fn_prime.to_numpy()
)

# TODO report value

expected_q = (policy * q_prime).sum(dim="action")

performance_difference = (1 / (1 - gamma)) * np.sum(
    norm_state_occupancy * (expected_q - value_prime)
)

# TODO report value
