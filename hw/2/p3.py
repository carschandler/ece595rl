import numpy as np

# p = np.array(
#     [
#         [
#             [1, 0],
#             [0, 1],
#         ],
#         [
#             [0.2, 0.8],
#             [0.6, 0.4],
#         ],
#     ]
# )


states = list("bc")
actions = list("xy")


def p_state(s, a):
    match (s, a):
        case ("b", "x"):
            return np.array([1.0, 0.0])
        case ("b", "y"):
            return np.array([0.2, 0.8])
        case ("c", "x"):
            return np.array([0.0, 1.0])
        case ("c", "y"):
            return np.array([0.6, 0.4])
        case _:
            raise Exception()


def reward(s, a):
    match (s, a):
        case ("b", "x"):
            return 0
        case ("b", "y"):
            return 0
        case ("c", "x"):
            return 1
        case ("c", "y"):
            return 1
        case _:
            raise Exception()


def opt_value(value_current, n_timesteps=1):
    value_next = None

    for _ in range(n_timesteps):
        if value_next is not None:
            value_current = value_next

        value_next = dict(b=None, c=None)

        # For a given state, we need to look at each possible action and return the the
        # maximum for that state
        for s in list("bc"):
            for a in list("xy"):
                print(f"P(s={s}, a={a}) = {p_state(s, a)}")
                ret = reward(s, a) + np.sum(
                    p_state(s, a) * np.array(list(value_current.values()))
                )
                print(f"  return = {ret}")
                if value_next[s] is None or ret > value_next[s]:
                    value_next[s] = ret

        print()

    return value_next


print(opt_value(dict(b=10, c=5)))

print(opt_value(dict(b=5, c=15)))
