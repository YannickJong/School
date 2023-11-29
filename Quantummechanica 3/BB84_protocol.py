import numpy as np


def print_for_excel(arr: np.ndarray):
    for i in range(arr.shape[0]):
        print(arr[i], end="\n")
    return None


np.random.seed(425)
Oa = np.random.choice(["x", "z"], size=30)
S = np.random.choice(["a", "b"], size=30)
M = np.where(S == "a", 0, 1)

Ob = np.random.choice(["x", "z"], size=30)
print_for_excel(Ob)
R = np.where(Ob == Oa, M, np.random.choice([0, 1]))

Aar = np.where(Oa == Ob, "a", "r")
Bar = np.where(Ob == Oa, "a", "r")

Ka = np.where(Aar == "a", M, "-")
Kb = np.where(Bar == "a", R, "-")

Ka = Ka[Ka != "-"]
Kb = Kb[Kb != "-"]

shared_key_a = Ka[np.arange(1, Ka.shape[0], 2)]
shared_key_b = Kb[np.arange(1, Kb.shape[0], 2)]

if np.array_equal:
    print("No eavesdropper")
else:
    print("Eavesdropper detected")
            