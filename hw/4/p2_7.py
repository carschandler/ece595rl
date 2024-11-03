import numpy as np
from IPython.display import Markdown
from p2_1 import value_fn
from p2_6 import vhat

value_error = np.abs(vhat - value_fn)[0]

value_error_latex = Markdown(f"{value_error:.5f}")
