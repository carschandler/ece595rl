import itertools

import numpy as np
import xarray as xr
from IPython.display import Markdown

state_space = np.arange(-3, 4)
action_space = np.arange(-1, 2)


def qhat0(s, a):
    return 2 * s**2 + a**2 - s * a + 0.5


# Initialize qhat0
q_eval = xr.DataArray(
    data=np.zeros([len(state_space), len(action_space)]),
    coords={"s": state_space, "a": action_space},
)

# Evaluate qhat0 at all (s, a)
for s, a in itertools.product(state_space, action_space):
    q_eval.loc[dict(s=s, a=a)] = qhat0(s, a)

# Get the optimal actions for qhat0
i_optimal_action = q_eval.argmax(dim="a").to_numpy()  # type: ignore

optimal_actions = action_space[i_optimal_action]

# Uniform distribution
p0 = xr.full_like(q_eval, 1 / 3)

pbar = xr.zeros_like(q_eval)

# We calculated optimal actions, so we "pick" the action accordingly (assign it
# probability 1)
for s, a in zip(state_space, optimal_actions):
    pbar.loc[dict(s=s, a=a)] = 1

alpha = 0.25

# Convex combination of policies
p1 = (1 - alpha) * p0 + alpha * pbar

# Get the distributions for s = 1, 2
p1_s1 = Markdown(str(p1.sel(s=1).data.tolist()))
p1_s2 = Markdown(str(p1.sel(s=2).data.tolist()))
