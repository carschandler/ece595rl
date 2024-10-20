import numpy as np
from IPython.display import Markdown
from p4_1 import gammas, trajectories

vhat = {state: 0.0 for state in "bcde"}
num_visits = {state: 0.0 for state in "bcde"}
returns = {state: [] for state in "bcde"}


def return_at_t(t, traj, gammas):
    return float(np.sum(traj["rewards"][t:] * gammas[t:]))


def mc_evaluation(vhat, returns, num_visits, return_function):
    for traj in trajectories:
        # For each state-action pair... (excluding state c since we end at that state)
        for state in np.unique(traj["states"][:-1]):
            # For each step in the trajectory...
            for t, (s, a, r) in enumerate(
                zip(traj["states"][:-1], traj["actions"], traj["rewards"])
            ):
                # We only want to evaluate each time where state is visited
                if s != state:
                    continue

                ret = return_function(t, traj, gammas)

                returns[state].append(ret)

                num_visits[state] += 1

            vhat[state] = float(
                (1 / num_visits[state]) * np.sum(returns[state])
            )

    return vhat, returns, num_visits


mc_evaluation(vhat, returns, num_visits, return_function=return_at_t)

vhat_latex = Markdown(
    r"\\".join(
        [rf"V^{{\pi^b}}({s}) &\approx {v:.5f}" for s, v in vhat.items()]
    )
)
