from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import mujoco
import mujoco.viewer
import csv
import os

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

MIN_STEPS_BEFORE_CRASH = 100

# ── SUCCESS SETTINGS ──
TARGET_THRESHOLD = 0.05
HOVER_TIME_REQUIRED = 3.0  # seconds

# Logging / saving
LOG_FILE = Path("training_log.csv")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Policy Network
# ─────────────────────────────────────────────

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 128),
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
    # gyro = data.sensor("body_gyro").data / 10.0

    return np.concatenate([
        pos, vel, quat,
        # gyro,
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
# Save model
# ─────────────────────────────────────────────

def save_model(policy, episode):
    torch.save(policy.state_dict(), CHECKPOINT_DIR / "latest.pt")
    if episode % 50 == 0:
        torch.save(policy.state_dict(), CHECKPOINT_DIR / f"episode_{episode}.pt")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    gains = PIDGains()

    timestep = model.opt.timestep
    hover_steps_required = int(HOVER_TIME_REQUIRED / timestep)

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "steps"])
    print("TIMESTEP:", model.opt.timestep)
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

            # ── SUCCESS TRACKING ──
            success_counter = 0

            for t in range(EPISODE_LENGTH):

                viewer.sync()

                obs_t = torch.tensor(obs, dtype=torch.float32)

                mean, std = policy(obs_t)
                dist = torch.distributions.Normal(mean, std)

                u_nn = dist.sample()
                log_prob = dist.log_prob(u_nn).sum()

                u_pid = compute_pid_control(
                    data, sp, gains, timestep, state=state
                )

                action = u_pid + ACTION_SCALE * u_nn.detach().numpy()
                action = np.clip(action, [0, -1, -1, -1], [1, 1, 1, 1])

                data.ctrl[:] = action
                mujoco.mj_step(model, data)

                reward = compute_reward(data, sp, action)

                # ── SUCCESS CONDITION (hover stability) ──
                pos = data.qpos[:3]
                target = np.array([sp.x, sp.y, sp.z])
                pos_err = np.linalg.norm(pos - target)

                if pos_err < TARGET_THRESHOLD:
                    success_counter += 1
                else:
                    success_counter = 0

                if success_counter >= hover_steps_required:
                    reward += 20.0
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    break

                # ── CRASH CONDITION ──
                if pos[2] < 0.05 and t > MIN_STEPS_BEFORE_CRASH:
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
                loss += -log_prob * G # REINFORCE loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward = sum(rewards)

            save_model(policy, episode)

            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode, total_reward, len(rewards)])

            print(f"Episode {episode:04d} | Reward: {total_reward:8.2f} | Steps: {len(rewards)}")


if __name__ == "__main__":
    main()