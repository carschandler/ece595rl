import numpy as np
import xarray as xr
from IPython.display import Markdown

state_space = [1, 2, 3]
action_space = list("gh")

# Define a multidimensional array containing probabilities of transitions from
# one state-action pair to another. This is just like a regular numpy array but with labeled "coordinates"
# and axes which make it easy for us to index it for a given state and/or
# action.
trans_fn = xr.DataArray(
    data=np.array(
        [
            [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]],
            [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
            [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
        ]
    ),
    coords=dict(
        current_state=state_space, action=action_space, next_state=state_space
    ),
)

# TODO verify below
# Probability of transitioning to each state across all state-action
# combinations should be 1
assert np.allclose(1, trans_fn.sum(axis="next_state"))

reward_fn = xr.DataArray(
    np.zeros([3, 2]),
    coords=dict(state=state_space, action=action_space),
)
reward_fn.loc[dict(state=3, action="h")] = 1

gamma = 0.95

# gammas = gamma ** np.arange(5)
#
# returns = [np.sum(traj["rewards"] * gammas) for traj in trajectories]
#
# returns_latex = Markdown(
#     r"\\".join([f"G_{i+1} &= {ret:0.4f}" for i, ret in enumerate(returns)])
# )
#
# policy_t = xr.DataArray(
#     data=np.array(
#         [
#             [0.1, 0.1, 0.8],
#             [0.05, 0.2, 0.75],
#             [0.25, 0.5, 0.25],
#             [0.9, 0.02, 0.08],
#         ]
#     ),
#     coords=dict(state=list("bcde"), action=list("xyz")),
# )
#
# policy_b = xr.DataArray(
#     data=np.array(
#         [
#             [0.5, 0.1, 0.4],
#             [0.05, 0.2, 0.75],
#             [0.5, 0.2, 0.3],
#             [0.9, 0.02, 0.08],
#         ]
#     ),
#     coords=dict(state=list("bcde"), action=list("xyz")),
# )
#
#
# # Double-check that our probabilities for a given state all sum to one
# assert np.allclose(policy_t.sum(dim="action"), 1)
# assert np.allclose(policy_b.sum(dim="action"), 1)
#
# p_target = {2: 0.0, 4: 0.0}
# p_behavior = {2: 0.0, 4: 0.0}
#
#
# for prob, policy in zip([p_target, p_behavior], [policy_t, policy_b]):
#     # Walk each step of the trajectory, determine the probability of taking an
#     # action at a given state according to the policy of interest, and take the
#     # product of all the probabilities to get the total probability of the
#     # trajectory.
#     for traj, i_traj in zip([trajectories[1], trajectories[3]], [2, 4]):
#         prob[i_traj] = float(
#             np.prod(
#                 [
#                     policy.sel(state=s, action=a)
#                     # We included the final state in the definition, but we
#                     # don't need it here, so we skip it in our iteration (i.e.
#                     # traj["states"[:-1]])
#                     for s, a in zip(traj["states"][:-1], traj["actions"])
#                 ]
#             )
#         )
#
# vhat = {state: 0.0 for state in "bcde"}
# num_visits = {state: 0.0 for state in "bcde"}
# returns = {state: [] for state in "bcde"}
#
#
# def return_at_t(t, traj, gammas):
#     return float(np.sum(traj["rewards"][t:] * gammas[t:]))
#
#
# def mc_evaluation(vhat, returns, num_visits, return_function):
#     for traj in trajectories:
#         # For each state-action pair... (excluding state c since we end at that state)
#         for state in np.unique(traj["states"][:-1]):
#             # For each step in the trajectory...
#             for t, (s, a, r) in enumerate(
#                 zip(traj["states"][:-1], traj["actions"], traj["rewards"])
#             ):
#                 # We only want to evaluate each time where state is visited
#                 if s != state:
#                     continue
#
#                 ret = return_function(t, traj, gammas)
#
#                 returns[state].append(ret)
#
#                 num_visits[state] += 1
#
#             vhat[state] = float(
#                 (1 / num_visits[state]) * np.sum(returns[state])
#             )
#
#     return vhat, returns, num_visits
#
#
# mc_evaluation(vhat, returns, num_visits, return_function=return_at_t)
#
# vhat_latex = Markdown(
#     r"\\".join(
#         [rf"V^{{\pi^b}}({s}) &\approx {v:.5f}" for s, v in vhat.items()]
#     )
# )
#
# vhat = {state: 0.0 for state in "bcde"}
# num_visits = {state: 0.0 for state in "bcde"}
# returns = {state: [] for state in "bcde"}
#
#
# def weighted_return_at_t(t, traj, gammas):
#     weight = np.prod(
#         [
#             (
#                 policy_t.sel(state=s, action=a)
#                 / policy_b.sel(state=s, action=a)
#             )
#             for s, a in zip(traj["states"][:-1][t:], traj["actions"][t:])
#         ]
#     )
#
#     return weight * return_at_t(t, traj, gammas)
#
#
# mc_evaluation(vhat, returns, num_visits, return_function=weighted_return_at_t)
#
# vhat_latex = Markdown(
#     r"\\".join(
#         [rf"V^{{\pi^b}}({s}) &\approx {v:.5f}" for s, v in vhat.items()]
#     )
# )
