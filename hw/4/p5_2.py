from IPython.display import Markdown
from p1_1 import gamma
from p5_1 import policy, q_prime, value_prime
from p2_4 import norm_state_occupancy
import numpy as np

expected_q = (policy * q_prime).sum(dim="action")

performance_difference = (1 / (1 - gamma)) * np.sum(
    norm_state_occupancy * (expected_q - value_prime)
)

performance_difference_latex = Markdown(f"{performance_difference:.5f}")
