import numpy as np
from IPython.display import Markdown
from p1_1 import gamma
from p2_1 import expected_trans_fn

norm_state_occupancy = (
    (1 - gamma)
    * np.linalg.inv(np.identity(3) - gamma * expected_trans_fn.T)
    @ [1, 0, 0]
)

assert np.allclose(1, norm_state_occupancy.sum())

norm_state_occupancy_latex = Markdown(
    r"\begin{matrix}"
    + ",&".join([f"{v:.5f}" for v in norm_state_occupancy])
    + r"\end{matrix}"
)
