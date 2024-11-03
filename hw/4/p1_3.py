import numpy as np
import xarray as xr
from IPython.display import Markdown
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
    trajectories.append(dict(states=states, actions=actions, rewards=rewards))


gammas = gamma ** np.arange(t_effective)

vhat = {state: 0.0 for state in state_space}
num_visits = {state: 0.0 for state in state_space}
returns = {state: [] for state in state_space}


def return_at_t(t, traj, gammas):
    return float(np.sum(traj["rewards"][t:] * gammas[: len(gammas) - t]))


def weighted_return_at_t(t, traj, gammas):
    weight = np.prod(
        [
            (
                policy["target"].sel(state=s, action=a)
                / policy["behavior"].sel(state=s, action=a)
            )
            for s, a in zip(traj["states"][t:], traj["actions"][t:])
        ]
    )

    return weight * return_at_t(t, traj, gammas)


def mc_evaluation(vhat, returns, num_visits, return_function):
    state = 1
    for traj in trajectories:
        # For each step in the trajectory...
        for t, s in enumerate(traj["states"]):
            # We only want to evaluate the time where state is visited
            if s != state:
                continue

            ret = return_function(t, traj, gammas)
            num_visits[state] += 1

            returns[state].append(ret)

            # We only want the first visit
            break

        vhat[state] = float((1 / num_visits[state]) * np.sum(returns[state]))

    return vhat, returns, num_visits


mc_evaluation(vhat, returns, num_visits, return_function=weighted_return_at_t)


vhat_latex = Markdown(rf"V^{{\pi^t}}(1) \approx {vhat[1]:.5f} \\")
