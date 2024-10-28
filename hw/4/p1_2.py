import numpy as np
from p1_1 import gamma, reward_fn

epsilon = 0.1

t_effective = int(
    np.ceil(
        np.log(reward_fn.max() / (epsilon * (1 - gamma))) / np.log(1 / gamma)
    )
)
