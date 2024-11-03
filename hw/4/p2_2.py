import itertools

import numpy as np
import xarray as xr
from IPython.display import Markdown
from p1_1 import action_space, state_space, trans_fn

NUM_SAMPLES = 100

phat = xr.zeros_like(trans_fn)

rng = np.random.default_rng(seed=42)

for s, a in itertools.product(state_space, action_space):
    distribution = trans_fn.sel(current_state=s, action=a)

    samples = rng.choice(state_space, p=distribution, size=NUM_SAMPLES)

    phat.loc[dict(current_state=s, action=a)] = np.array(
        [np.sum(samples == s) / NUM_SAMPLES for s in state_space]
    )

phat_latex = Markdown(
    r"\begin{matrix}"
    + r"\\".join(
        [
            ",&".join(
                rf"\hat{{P}}({v.next_state.item()} |"
                rf" {v.current_state.item()},"
                f" {v.action.item()}) = {v.item():.2f}"
                for v in v_sn
            )
            for v_sn in phat.stack(
                {"sa": ["current_state", "action"], "sn": ["next_state"]}
            )
        ]
    )
    + r"\end{matrix}"
)
