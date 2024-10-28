import numpy as np
import xarray as xr
from p1_1 import action_space, gamma, policy, reward_fn, state_space, trans_fn
from p1_2 import t_effective

NUM_TRAJECTORIES = 50

rng = np.random.default_rng(seed=42)


def sample_trajectory(
    trans_fn: xr.DataArray, policy: xr.DataArray, reward_fn: xr.DataArray
):
    states = []
    actions = []
    rewards = []

    state = 1

    for _ in range(t_effective):
        action = str(rng.choice(action_space, p=policy.sel(state=state)))
        reward = float(reward_fn.sel(state=state, action=action))

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = int(
            rng.choice(
                state_space,
                p=trans_fn.sel(current_state=state, action=action),
            )
        )

    return (np.array(vals) for vals in [states, actions, rewards])


trajectories = []
for _ in range(NUM_TRAJECTORIES):
    states, actions, rewards = sample_trajectory(
        trans_fn, policy["behavior"], reward_fn
    )
    trajectories.append(dict(s=states, a=actions, r=rewards))


gammas = gamma ** np.arange(t_effective)

returns = [np.sum(traj["r"] * gammas) for traj in trajectories]


#
# returns_latex = Markdown(
#     r"\\".join([f"G_{i+1} &= {ret:0.4f}" for i, ret in enumerate(returns)])
# )
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
