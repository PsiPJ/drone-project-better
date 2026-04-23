from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import mujoco
import mujoco.viewer

from pid_control import compute_pid_control, PIDGains, PIDState, Setpoint

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

SCENE = Path(__file__).resolve().parent / "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml"

EPISODE_LENGTH = 1000
NUM_EPISODES = 5000
GAMMA = 0.99
LR = 3e-4

ACTION_SCALE = 0.3

# ─────────────────────────────────────────────
# Policy Network
# ─────────────────────────────────────────────

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, 4)

        self.log_std = nn.Parameter(torch.full((4,), -0.5))

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)

        log_std = torch.clamp(self.log_std, -2, 1)
        std = torch.exp(log_std)

        return mean, std


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

def get_obs(data: mujoco.MjData, sp: Setpoint):
    pos = data.qpos[:3] / 2.0
    vel = data.qvel[:3] / 5.0

    quat = data.sensor("body_quat").data
    gyro = data.sensor("body_gyro").data / 10.0

    return np.concatenate([
        pos, vel, quat, gyro,
        np.array([sp.x, sp.y, sp.z, sp.yaw], dtype=np.float32) / 2.0
    ]).astype(np.float32)


# ─────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────

def compute_reward(data: mujoco.MjData, sp: Setpoint, action):
    pos = data.qpos[:3]
    vel = data.qvel[:3]

    target = np.array([sp.x, sp.y, sp.z])
    pos_err = np.linalg.norm(pos - target)

    reward = 1.0 - pos_err
    reward -= 0.1 * np.linalg.norm(vel)
    reward -= 0.01 * np.linalg.norm(action)

    if pos_err < 0.05:
        reward += 5.0

    if pos[2] < 0.2:
        reward -= 10.0

    return reward


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    gains = PIDGains()

    # IMPORTANT: viewer wraps full training loop
    with mujoco.viewer.launch_passive(model, data) as viewer:

        for episode in range(NUM_EPISODES):

            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)

            data.ctrl[:] = [0.3, 0, 0, 0]
            for _ in range(10):
                mujoco.mj_step(model, data)

            data.qpos[:3] += np.random.uniform(-0.1, 0.1, 3)

            sp = Setpoint(
                x=np.random.uniform(-0.5, 0.5),
                y=np.random.uniform(-0.5, 0.5),
                z=np.random.uniform(0.8, 1.2),
                yaw=0.0
            )

            state = PIDState()

            log_probs = []
            rewards = []

            obs = get_obs(data, sp)

            for t in range(EPISODE_LENGTH):

                # ── viewer update ──
                viewer.sync()

                obs_t = torch.tensor(obs, dtype=torch.float32)

                mean, std = policy(obs_t)
                dist = torch.distributions.Normal(mean, std)

                u_nn = dist.sample()
                log_prob = dist.log_prob(u_nn).sum()

                u_pid = compute_pid_control(
                    data, sp, gains, model.opt.timestep, state=state
                )

                action = u_pid + ACTION_SCALE * u_nn.detach().numpy()
                action = np.clip(action, [0, -1, -1, -1], [1, 1, 1, 1])

                data.ctrl[:] = action
                mujoco.mj_step(model, data)

                reward = compute_reward(data, sp, action)

                if data.qpos[2] < 0.05 and t > 20:
                    reward -= 50.0
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    break

                log_probs.append(log_prob)
                rewards.append(reward)

                obs = get_obs(data, sp)

            # ── update policy ──
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)

            returns = torch.tensor(returns, dtype=torch.float32)

            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            else:
                returns = torch.zeros_like(returns)

            loss = 0
            for log_prob, G in zip(log_probs, returns):
                loss += -log_prob * G

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Episode {episode:04d} | Reward: {sum(rewards):8.2f} | Steps: {len(rewards)}")


if __name__ == "__main__":
    main()