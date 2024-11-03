import numpy as np
from IPython.display import Markdown
from p1_1 import gamma, reward_fn
from p2_3 import l1_error
from p2_4 import norm_state_occupancy

simulation_lemma_bound = (
    (gamma * np.max(reward_fn))
    / (1 - gamma) ** 2
    * np.sum(norm_state_occupancy * l1_error)
)


simulation_lemma_bound_latex = Markdown(f"{simulation_lemma_bound:.5f}")
