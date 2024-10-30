import itertools

import numpy as np
import xarray as xr

state_space = [1, 2, 3]
action_space = list("gh")

# Define a multidimensional array containing probabilities of transitions from
# one state-action pair to another. This is just like a regular numpy array but with labeled "coordinates"
# and axes which make it easy for us to index it for a given state and/or
# action.
trans_fn = xr.DataArray(
    data=np.array(
        [
            [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]],
            [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
            [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
        ]
    ),
    coords=dict(
        current_state=state_space, action=action_space, next_state=state_space
    ),
)

# For any current state-action pair, the sum of transition probabilities to all
# next states should sum to 1
assert np.allclose(1, trans_fn.sum(dim="next_state"))

reward_fn = xr.DataArray(
    np.zeros([3, 2]),
    coords=dict(state=state_space, action=action_space),
)
reward_fn.loc[dict(state=3, action="h")] = 1

gamma = 0.95

policy = {1: "g", 2: "g", 3: "h"}

expected_trans_fn = np.reshape(
    [
        trans_fn.sel(current_state=s, next_state=sn, action=policy[s])
        for s, sn in itertools.product(state_space, state_space)
    ],
    (3, 3),
)

expected_reward_fn = np.array(
    [reward_fn.sel(state=s, action=policy[s]) for s in state_space]
)

value_fn = (
    np.linalg.inv(np.identity(len(state_space)) - gamma * expected_trans_fn)
    @ expected_reward_fn
)


# TODO: latex?

print(f"Value function at each state: {value_fn}")
