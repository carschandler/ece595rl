import itertools

import numpy as np
import xarray as xr
from p2_1 import (
    action_space,
    expected_reward_fn,
    expected_trans_fn,
    gamma,
    policy,
    reward_fn,
    state_space,
    trans_fn,
    value_fn,
)

NUM_SAMPLES = 100

phat = xr.zeros_like(trans_fn)

rng = np.random.default_rng(seed=42)

for s, a in itertools.product(state_space, action_space):
    distribution = trans_fn.sel(current_state=s, action=a)

    samples = rng.choice(state_space, p=distribution, size=NUM_SAMPLES)

    phat.loc[dict(current_state=s, action=a)] = np.array(
        [np.sum(samples == s) / NUM_SAMPLES for s in state_space]
    )

print("TODO: report estimated phat")

## p2-3

l1_error = np.array(
    [
        np.linalg.norm(
            (phat - trans_fn).sel(current_state=s, action=policy[s]), ord=1
        )
        for s in state_space
    ]
)

print("TODO: report l1 error")

norm_state_occupancy = (
    (1 - gamma)
    * np.linalg.inv(np.identity(3) - gamma * expected_trans_fn.T)
    @ [1, 0, 0]
)

assert np.allclose(1, norm_state_occupancy.sum())

print("TODO: report rhobar")

simulation_lemma_bound = (
    (gamma * np.max(reward_fn))
    / (1 - gamma) ** 2
    * np.sum(norm_state_occupancy * l1_error)
)

print("TODO: report simulation_lemma_bound")

expected_phat = np.reshape(
    [
        phat.sel(current_state=s, next_state=sn, action=policy[s])
        for s, sn in itertools.product(state_space, state_space)
    ],
    (3, 3),
)


vhat = (
    np.linalg.inv(np.identity(len(state_space)) - gamma * expected_phat)
    @ expected_reward_fn
)

print("TODO: report vhat")

value_error = np.abs(vhat - value_fn)[0]

print("TODO: report vhat error")
