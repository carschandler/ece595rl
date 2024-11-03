from IPython.display import Markdown
from p1_1 import value_fn
from p1_3 import vhat

# NOTE: vhat is a dict; these indices aren't mismatched
value_error = vhat[1] - value_fn["target"][0]

value_error_latex = Markdown(
    rf"\hat{{V}}^{{\pi^t}}(1) - V^{{\pi^t}}(1) = {vhat[1]:.5f} -"
    rf" {value_fn['target'][0]:.5f} = {value_error:.5f}"
)
