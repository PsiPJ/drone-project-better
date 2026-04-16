import mujoco
import numpy as np
import mujoco.viewer
import time

# ============================================================
# 🧠 NEURAL NETWORK IMPORTS
# ============================================================
# PyTorch is used here to define a neural network policy
# that will eventually replace or augment the hand-written controller.
import torch
import torch.nn as nn


# ============================================================
# 🧠 NEURAL NETWORK POLICY (DRONE BRAIN)
# ============================================================
# This is a simple feedforward neural network.
# Input: drone state (position, velocity, orientation, etc.)
# Output: control signals (thrust, roll, pitch, yaw)

class DronePolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # Fully connected neural network
        self.net = nn.Sequential(
            nn.Linear(17, 64),   # input: state vector (17 values)
            nn.ReLU(),           # nonlinearity
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),    # output: 4 control signals
            nn.Tanh()            # outputs bounded between [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 🧠 LOAD MUJOCO SIMULATION
# ============================================================
# This loads the Crazyflie drone simulation model.
model = mujoco.MjModel.from_xml_path(
    "mujoco_menagerie/bitcraze_crazyflie_2/scene.xml"
)
data = mujoco.MjData(model)

# Create neural network policy instance
policy = DronePolicy()
policy.eval()  # sets network to inference mode (no training behavior)


# ============================================================
# 🧠 MAIN SIMULATION LOOP
# ============================================================
# This loop continuously:
# 1. reads sensors
# 2. computes control actions
# 3. applies them to the drone
# 4. advances simulation

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        step_start = time.time()

        # ====================================================
        # 📡 SENSOR READINGS (REAL-TIME DRONE STATE)
        # ====================================================

        # Position in world frame (x, y, z)
        current_position = data.qpos[:3]

        # Linear velocity in world frame (vx, vy, vz)
        current_velocity = data.qvel[:3]

        # Orientation as quaternion (w, x, y, z)
        attitude_quat = data.sensor("body_quat").data

        # Angular velocity (rotation speed around axes)
        angular_velocity = data.sensor("body_gyro").data


        # ====================================================
        # 🎯 USER-DEFINED GOAL
        # ====================================================
        # This is where you tell the drone what "success" means.

        target_x, target_y, target_z = 0.8, 0.0, 1.0
        goal_yaw = 0.0


        # ====================================================
        # 🧠 BUILD STATE VECTOR FOR NEURAL NETWORK
        # ====================================================
        # The neural network needs everything in one vector.

        state = np.concatenate([
            current_position,     # where drone is
            current_velocity,     # how it's moving
            attitude_quat,        # orientation
            angular_velocity,     # rotation speed
            np.array([target_x, target_y, target_z, goal_yaw])
        ])


        # ====================================================
        # 🧠 CONVERT STATE → TENSOR (FOR PYTORCH)
        # ====================================================
        # Neural networks only understand tensors.

        state_tensor = torch.tensor(state, dtype=torch.float32)


        # ====================================================
        # 🧠 NEURAL NETWORK FORWARD PASS
        # ====================================================
        # The policy predicts control outputs from the state.

        nn_action = policy(state_tensor).detach().numpy()

        # Convert network output to usable control signals

        thrust_action = (nn_action[0] + 1) / 2  # scale [-1,1] → [0,1]
        roll_action   = nn_action[1]
        pitch_action  = nn_action[2]
        yaw_action    = nn_action[3]


        # ====================================================
        # ⚙️ APPLY CONTROL INPUTS TO DRONE
        # ====================================================
        # These directly control the physics simulation.

        data.ctrl[:] = [
            thrust_action,
            roll_action,
            pitch_action,
            yaw_action
        ]


        # ====================================================
        # ⏱ STEP SIMULATION FORWARD
        # ====================================================
        mujoco.mj_step(model, data)
        viewer.sync()

        # Keep simulation real-time (prevents speed-up/slow-down)
        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))