from copy import deepcopy

import numpy as np
import xarray as xr

# Define a multidimensional array containing probabilities of state-action
# pairs. This is just like a regular numpy array but with labeled "coordinates"
# and axes which make it easy for us to index it for a given state and/or
# action.
policy_t = xr.DataArray(
    data=np.array(
        [
            [0.1, 0.1, 0.8],
            [0.05, 0.2, 0.75],
            [0.25, 0.5, 0.25],
            [0.9, 0.02, 0.08],
        ]
    ),
    coords=dict(state=list("bcde"), action=list("xyz")),
)

policy_b = xr.DataArray(
    data=np.array(
        [
            [0.5, 0.1, 0.4],
            [0.05, 0.2, 0.75],
            [0.5, 0.2, 0.3],
            [0.9, 0.02, 0.08],
        ]
    ),
    coords=dict(state=list("bcde"), action=list("xyz")),
)


# Double-check that our probabilities for a given state all sum to one
assert np.allclose(policy_t.sum(dim="action"), 1)
assert np.allclose(policy_b.sum(dim="action"), 1)

from p4_1 import trajectories

p_target = {2: 0.0, 4: 0.0}
p_behavior = {2: 0.0, 4: 0.0}


for prob, policy in zip([p_target, p_behavior], [policy_t, policy_b]):
    # Walk each step of the trajectory, determine the probability of taking an
    # action at a given state according to the policy of interest, and take the
    # product of all the probabilities to get the total probability of the
    # trajectory.
    for traj, i_traj in zip([trajectories[1], trajectories[3]], [2, 4]):
        prob[i_traj] = float(
            np.prod(
                [
                    policy.sel(state=s, action=a)
                    # We included the final state in the definition, but we
                    # don't need it here, so we skip it in our iteration (i.e.
                    # traj["states"[:-1]])
                    for s, a in zip(traj["states"][:-1], traj["actions"])
                ]
            )
        )
