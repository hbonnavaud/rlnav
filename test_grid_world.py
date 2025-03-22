import os
import gymnasium as gym
import imageio
import numpy as np
from rlnav import GridWorld
import matplotlib.pyplot as plt

# Create environment
env = GridWorld(goal_conditioned=True)
env.reset()

plt.imshow(env.render())
plt.show()