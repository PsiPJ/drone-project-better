import torch
import torch.nn as nn
import torch.nn.functional as F
import mujoco
import numpy as np
import mujoco.viewer
import time

class Actor(nn.Module):
    """Policy Network: State -> Action"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, action_dim) # Mean of action
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) # Exploration

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.mu(x)) # Normalized actions [-1, 1]

class Critic(nn.Module):
    """Value Network: State -> Expected Reward"""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)


# 1. Load your model
model = mujoco.MjModel.from_xml_path('mujoco_menagerie/bitcraze_crazyflie_2/scene.xml')
data = mujoco.MjData(model)

# 2. Launch the viewer
# The 'with' block ensures the window closes cleanly when done
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the window or stop at 10 seconds
    while viewer.is_running() and data.time < 10.0:
        step_start = time.time()

        # [Insert your Brain/Neural Network logic here]
        # Example: Applying your hover thrust
        data.ctrl[:] = 0.26487 

        # Step the physics
        mujoco.mj_step(model, data)

        # 3. Sync the viewer 
        # This is what actually updates the frame in the window
        viewer.sync()

        # (Optional) Slow down the loop to real-time so it's not a blur
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)