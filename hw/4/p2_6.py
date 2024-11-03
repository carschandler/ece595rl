import itertools

import numpy as np
from IPython.display import Markdown
from p1_1 import gamma, state_space
from p2_1 import expected_reward_fn, policy
from p2_2 import phat

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

vhat_latex = Markdown(
    r"\\".join(
        [
            rf"\hat{{V}}^{{\pi}}({s}) = {v:.5f}"
            for s, v in zip(list("123"), vhat)
        ]
    )
)
