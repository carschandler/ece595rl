import numpy as np
from IPython.display import Markdown

trajectories = [
    dict(
        states=np.array(list("cbdeec")),
        actions=np.array(list("zxyxx")),
        rewards=np.array([1, 0, 1, 2, 2]),
    ),
    dict(
        states=np.array(list("cdecee")),
        actions=np.array(list("yxzzx")),
        rewards=np.array([0, 1, -1, 1, 2]),
    ),
    dict(
        states=np.array(list("cbcebe")),
        actions=np.array(list("yzyxy")),
        rewards=np.array([0, 0.5, 0, 2, 0]),
    ),
    dict(
        states=np.array(list("ccebed")),
        actions=np.array(list("zxxyy")),
        rewards=np.array([1, 0.5, 2, 0, -1]),
    ),
]

gamma = 0.9

gammas = gamma ** np.arange(5)

returns = [np.sum(traj["rewards"] * gammas) for traj in trajectories]

returns_latex = Markdown(
    r"\\".join([f"G_{i+1} &= {ret:0.4f}" for i, ret in enumerate(returns)])
)
