import numpy as np
import xarray as xr
from IPython.display import Markdown
from p1_1 import action_space, state_space

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

value_prime = (q_prime * policy_prime).sum(dim="action")

value_prime_latex = Markdown(
    r"\\".join(
        [
            rf"V^{{\pi}}({s}) = {v:.5f}"
            for s, v in zip(list("123"), value_prime)
        ]
    )
)
