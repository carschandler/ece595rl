import numpy as np
from IPython.display import Markdown
from p4_1 import gammas, trajectories
from p4_2 import policy_b, policy_t
from p4_4 import mc_evaluation, return_at_t

vhat = {state: 0.0 for state in "bcde"}
num_visits = {state: 0.0 for state in "bcde"}
returns = {state: [] for state in "bcde"}


def weighted_return_at_t(t, traj, gammas):
    weight = np.prod(
        [
            (
                policy_t.sel(state=s, action=a)
                / policy_b.sel(state=s, action=a)
            )
            for s, a in zip(traj["states"][:-1][t:], traj["actions"][t:])
        ]
    )

    return weight * return_at_t(t, traj, gammas)


mc_evaluation(vhat, returns, num_visits, return_function=weighted_return_at_t)

vhat_latex = Markdown(
    r"\\".join(
        [rf"V^{{\pi^b}}({s}) &\approx {v:.5f}" for s, v in vhat.items()]
    )
)
