Maps are list of list of integers.
They represent a plan on which the agent can navigate.
Here is the meaning of integers:
 - 0: empty tile,
 - 1: wall (unreachable area),
 - 2: agent start position (if reset_anywhere is false),
 - 3: rewarding state (if not goal-conditioned),
 - 4: trap! terminating state with negative reward.