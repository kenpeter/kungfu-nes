import numpy as np
from kungfu_env import KUNGFU_ACTIONS, KUNGFU_ACTION_NAMES  # Import actions from kungfu_env

data = np.load("recordings/KungFu-Nes_1Player.Level1_20250420-181321.npz")
actions = data["actions"]
print("Action counts:", np.bincount(actions))
print("Action names:", [KUNGFU_ACTION_NAMES[i] for i in range(len(np.bincount(actions)))])