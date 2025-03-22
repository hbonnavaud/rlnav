# NavigationRLEnvironments
A set of reinforcement learning environments for navigation tasks (Try to choose the best sequence of actions to move an agent to a given goal position).

## Environments

### GridEnv

### PointEnv

The agent is a point moving in continuous space.
In all the versions bellow, a gaussian noise is added to the action depending on a hyperparameter "noise_mean".


 - PointEnv-V1
   - actions: x, y modification of the agent current position.
   - observation: agent's x, y position.
 - PointEnv-V2
   - actions: angular and linear velocity. The agent rotate according to the given angular velocity, then move following the linear velocity.
   - observation: agent's x, y position and agent's orientation.
 - PointEnv-V3
   - actions: modification of the agent's angular and linear velocity. 
        The agent rotate according to it's angular velocity, then move following the linear velocity.
   - observation: agent's x, y position and agent's orientation.

### w
