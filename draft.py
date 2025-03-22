import importlib
import rlnav
import gymnasium

env = gymnasium.make("GridWorld-v0", map_name="medium_maze")  # ✅ Try again
env.reset()