import numpy as np
from p5_1 import pref_1_over_2, pref_1_over_3, r1, r2, r3


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


pref_3_over_1 = 1 - pref_1_over_3

mle_loss = -(
    pref_1_over_2 * np.log(sigmoid(r1 - r2))
    + pref_3_over_1 * np.log(sigmoid(r3 - r1))
)

mle_loss_2 = -(
    (1 - pref_1_over_2) * np.log(sigmoid(r2 - r1))
    + pref_3_over_1 * np.log(sigmoid(r3 - r1))
)
