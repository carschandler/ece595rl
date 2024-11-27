import numpy as np
import xarray as xr
from IPython.display import Markdown

reward = xr.DataArray(
    data=np.array(
        [
            [-1, 1, 0, 0, 0],
            [-2, 0, 0, 3, -5],
            [-2, -2, 5, 0, 0],
            [-2, -2, 4, 0, 0],
        ]
    ),
    coords=dict(
        state=["empty", "monster", "food", "resource"],
        action=["stay", "explore", "collect", "evade", "befriend"],
    ),
)

trajectories = [
    [("empty", "stay"), ("monster", "evade"), ("resource", "collect")],
    [("empty", "explore"), ("food", "collect"), ("monster", "befriend")],
    [("empty", "explore"), ("empty", "explore"), ("resource", "collect")],
]

gamma = 0.95


returns = np.sum(
    np.array(
        [
            [
                gamma**t * reward.sel(state=s, action=a).item()
                for t, (s, a) in enumerate(traj)
            ]
            for traj in trajectories
        ]
    ),
    axis=1,
)

r1, r2, r3 = returns

return_latex = Markdown(
    "\n".join([rf"\tau_{i+1}&: {r:0.4f} \\" for i, r in enumerate(returns)])
)

pref_1_over_2 = np.exp(r1) / np.sum([np.exp(r1), np.exp(r2)])
pref_1_over_3 = np.exp(r1) / np.sum([np.exp(r1), np.exp(r3)])
