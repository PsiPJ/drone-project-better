from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from pid_control import compute_pid_control, PIDGains, PIDState, Setpoint

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

SCENE = Path(__file__).resolve().parent / "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml"

NUM_EPISODES = 5000
GAMMA = 0.99
LR = 3e-4
CLIP_EPS = 0.2
PPO_EPOCHS = 4

ACTION_SCALE = 0.2
EPISODE_SECONDS = 8.0
TARGET_THRESHOLD = 0.05


# ─────────────────────────────────────────────
# ACTOR
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

        # smoother exploration
        std = torch.exp(torch.clamp(self.log_std, -2, 0.5))
        return mean, std


# ─────────────────────────────────────────────
# CRITIC
# ─────────────────────────────────────────────

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

def get_obs(data: mujoco.MjData, sp: Setpoint):
    pos = data.qpos[:3] / 2.0
    vel = data.qvel[:3] / 5.0
    quat = data.sensor("body_quat").data

    return np.concatenate([
        pos, vel, quat,
        np.array([sp.x, sp.y, sp.z, sp.yaw], dtype=np.float32) / 2.0
    ]).astype(np.float32)


# ─────────────────────────────────────────────
# STABILITY-FOCUSED REWARD (FIXED)
# ─────────────────────────────────────────────

def compute_reward(data: mujoco.MjData, sp: Setpoint, action, prev_action=None):
    pos = data.qpos[:3]
    vel = data.qvel[:3]

    target = np.array([sp.x, sp.y, sp.z])
    pos_err = np.linalg.norm(pos - target)

    # 🔥 prioritize stability over tracking
    reward = -0.3 * pos_err
    reward -= 0.6 * np.linalg.norm(vel)   # STRONG damping (key fix)
    reward -= 0.03 * np.linalg.norm(action)

    # jitter suppression (important for shake)
    if prev_action is not None:
        reward -= 0.4 * np.linalg.norm(action - prev_action)

    # stable hover bonus
    if pos_err < TARGET_THRESHOLD and np.linalg.norm(vel) < 0.05:
        reward += 1.2

    # crash penalty
    if pos[2] < 0.2:
        reward -= 5.0

    return reward


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    policy = Policy()
    value_fn = ValueNet()

    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()),
        lr=LR
    )

    gains = PIDGains()
    timestep = model.opt.timestep
    max_steps = int(EPISODE_SECONDS / timestep)

    episode_rewards = []

    try:
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

                obs_buf, act_buf, logp_buf = [], [], []
                reward_buf, value_buf = [], []

                obs = get_obs(data, sp)
                prev_action = None

                for t in range(max_steps):

                    viewer.sync()

                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                    mean, std = policy(obs_t)
                    dist = torch.distributions.Normal(mean, std)

                    action_delta = torch.tanh(dist.sample()).squeeze(0) * ACTION_SCALE

                    log_prob = dist.log_prob(action_delta / ACTION_SCALE).sum()
                    value = value_fn(obs_t).squeeze(0)

                    sp_rl = np.array([sp.x, sp.y, sp.z, sp.yaw]) + action_delta.detach().cpu().numpy()
                    new_sp = Setpoint(*sp_rl)

                    u_pid = compute_pid_control(
                        data, new_sp, gains, timestep, state=state
                    )

                    action = np.clip(u_pid, [0, -1, -1, -1], [1, 1, 1, 1])
                    data.ctrl[:] = action

                    mujoco.mj_step(model, data)

                    reward = compute_reward(data, sp, action, prev_action)

                    prev_action = action.copy()

                    obs_buf.append(obs)
                    act_buf.append(action_delta)
                    logp_buf.append(log_prob)
                    reward_buf.append(reward)
                    value_buf.append(value)

                    obs = get_obs(data, sp)

                # ─────────────────────────────
                # PPO UPDATE
                # ─────────────────────────────

                returns = []
                G = 0
                for r in reversed(reward_buf):
                    G = r + GAMMA * G
                    returns.insert(0, G)

                returns = torch.tensor(returns, dtype=torch.float32)
                values = torch.stack(value_buf).detach()

                advantages = returns - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32)
                act_t = torch.stack(act_buf)
                old_logp = torch.stack(logp_buf).detach()

                for _ in range(PPO_EPOCHS):

                    mean, std = policy(obs_t)
                    dist = torch.distributions.Normal(mean, std)

                    new_logp = dist.log_prob(act_t / ACTION_SCALE).sum(dim=1)

                    ratio = torch.exp(new_logp - old_logp)

                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages

                    actor_loss = -torch.min(surr1, surr2).mean()

                    value_pred = value_fn(obs_t)
                    value_loss = ((value_pred - returns) ** 2).mean()

                    loss = actor_loss + 0.5 * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                episode_rewards.append(sum(reward_buf))
                print(f"Episode {episode:04d} | Reward: {episode_rewards[-1]:8.2f}")

    finally:
        if len(episode_rewards) > 0:

            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards, label="Reward")

            if len(episode_rewards) > 10:
                smooth = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
                plt.plot(smooth, label="Smoothed")

            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Training Progress (Stable Control)")
            plt.legend()
            plt.grid()

            plt.savefig("reward_curve.png")
            print("Saved reward_curve.png")
            plt.show()


if __name__ == "__main__":
    main()