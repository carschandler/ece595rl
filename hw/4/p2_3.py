import numpy as np
from IPython.display import Markdown
from p1_1 import state_space, trans_fn
from p2_1 import policy
from p2_2 import phat

l1_error = np.array(
    [
        np.linalg.norm(
            (phat - trans_fn).sel(current_state=s, action=policy[s]), ord=1
        )
        for s in state_space
    ]
)

l1_error_latex = Markdown(
    r"\begin{matrix}"
    + ",&".join([f"{v:.5f}" for v in l1_error])
    + r"\end{matrix}"
)
