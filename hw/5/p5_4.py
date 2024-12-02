import numpy as np
from p5_1 import pref_1_over_2, pref_1_over_3

pref_3_over_1 = 1 - pref_1_over_3

mle_loss = -(np.log(pref_1_over_2) + np.log(pref_3_over_1))
