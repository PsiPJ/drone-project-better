import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from torch.distributions import Normal


# ============================================================
# 1. ACTOR-CRITIC NETWORK
# ============================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic        = nn.Linear(256, 1)

    def forward(self, state):
        features    = self.shared(state)
        action_mean = self.actor_mean(features)
        action_std  = torch.exp(self.actor_log_std.clamp(-2, 0.5))
        value       = self.critic(features)
        return action_mean, action_std, value

    def get_action(self, state):
        action_mean, action_std, value = self.forward(state)
        dist     = Normal(action_mean, action_std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value


# ============================================================
# 2. REWARD FUNCTION
# ============================================================
def compute_reward(data, target, prev_dist, crashed):
    pos  = data.qpos[:3]
    dist = np.linalg.norm(target - pos)

    proximity_reward = -dist
    progress_reward  = (prev_dist - dist) * 10.0
    alt_reward       = min(pos[2], target[2]) * 2.0
    hover_bonus      = 5.0 if dist < 0.1 else (1.0 if dist < 0.3 else 0.0)

    quat         = data.qpos[3:7]
    tilt_penalty = -abs(1.0 - quat[0]) * 3.0

    crash_penalty = -50.0 if crashed else 0.0

    total = proximity_reward + progress_reward + alt_reward + hover_bonus + tilt_penalty + crash_penalty
    return total, dist


# ============================================================
# 3. ROLLOUT BUFFER
# ============================================================
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones     = [], [], []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma=0.99, lam=0.95):
        returns, advantages = [], []
        gae    = 0
        values = self.values + [0]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae   = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return returns, advantages


# ============================================================
# 4. PPO TRAINER
# ============================================================
class PPOTrainer:
    def __init__(self, model, lr=1e-4, clip_eps=0.2, epochs=10, batch_size=64):
        self.model      = model
        self.optimizer  = optim.Adam(model.parameters(), lr=lr)
        self.clip_eps   = clip_eps
        self.epochs     = epochs
        self.batch_size = batch_size

    def update(self, buffer):
        returns, advantages = buffer.compute_returns()

        states  = torch.FloatTensor(np.array(buffer.states))
        actions = torch.FloatTensor(np.array(buffer.actions))
        old_lps = torch.FloatTensor(buffer.log_probs)
        returns = torch.FloatTensor(returns)
        advs    = torch.FloatTensor(advantages)

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        total_loss = 0
        for _ in range(self.epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                idx                          = indices[start:start + self.batch_size]
                sb, ab, lp_old, ret_b, adv_b = (
                    states[idx], actions[idx], old_lps[idx],
                    returns[idx], advs[idx]
                )

                action_mean, action_std, values = self.model(sb)
                dist          = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(ab).sum(dim=-1)
                entropy       = dist.entropy().sum(dim=-1).mean()

                ratio       = torch.exp(new_log_probs - lp_old)
                surr1       = ratio * adv_b
                surr2       = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), ret_b)
                loss        = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        buffer.clear()
        return total_loss


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================
def get_observation(data, target):
    pos          = data.qpos[:3]
    quat         = data.qpos[3:7]
    vel          = data.qvel[:3]
    ang_vel      = data.qvel[3:6]
    target_error = target - pos
    return np.concatenate([pos, quat, vel, ang_vel, target_error])


def reset_env(data_mj, model_mj):
    mujoco.mj_resetData(model_mj, data_mj)
    data_mj.qpos[2] = np.random.uniform(0.1, 0.3)
    mujoco.mj_forward(model_mj, data_mj)


# ============================================================
# 6. SETUP
# ============================================================
model_mj = mujoco.MjModel.from_xml_path('mujoco_menagerie/bitcraze_crazyflie_2/scene.xml')
data_mj  = mujoco.MjData(model_mj)

HOVER_THRUST  = 0.281
TARGET_POS    = np.array([0.0, 0.0, 1.0])
MAX_STEPS     = 2000
COLLECT_STEPS = 2048
CRASH_HEIGHT  = 0.02

state_dim  = 16
action_dim = 4

brain = ActorCritic(state_dim, action_dim)

if os.path.exists('drone_brain.pth'):
    brain.load_state_dict(torch.load('drone_brain.pth'))
    print("Loaded previous training checkpoint!")
else:
    print("Starting fresh training...")

trainer = PPOTrainer(brain)
buffer  = RolloutBuffer()


# ============================================================
# 7. MAIN TRAINING LOOP
# ============================================================
print("Starting PPO training. Close the viewer window to stop.")
print(f"Collecting {COLLECT_STEPS} steps per update.\n")

episode        = 0
total_steps    = 0
episode_steps  = 0
episode_reward = 0.0
prev_dist      = np.linalg.norm(TARGET_POS - np.array([0, 0, 0.2]))

reset_env(data_mj, model_mj)

with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # SENSE
        obs        = get_observation(data_mj, TARGET_POS)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        # THINK
        with torch.no_grad():
            action, log_prob, value = brain.get_action(obs_tensor)

        action_np = action.numpy()[0]

        # ACT
        thrust_adjustment = action_np * 0.1
        data_mj.ctrl[:]   = np.clip(HOVER_THRUST + thrust_adjustment, 0.0, 1.0)

        # STEP PHYSICS
        mujoco.mj_step(model_mj, data_mj)

        # REWARD & DONE
        crashed = data_mj.qpos[2] < CRASH_HEIGHT and episode_steps > 10
        done    = crashed or (episode_steps >= MAX_STEPS)

        reward, prev_dist = compute_reward(data_mj, TARGET_POS, prev_dist, crashed)
        episode_reward   += reward

        # STORE
        buffer.add(
            obs,
            action_np,
            log_prob.item(),
            reward,
            value.item(),
            float(done)
        )

        episode_steps += 1
        total_steps   += 1

        # RESET
        if done:
            episode += 1
            dist     = np.linalg.norm(TARGET_POS - data_mj.qpos[:3])
            print(f"Ep {episode:4d} | Steps {total_steps:7d} | "
                  f"Reward {episode_reward:8.1f} | "
                  f"Dist {dist:.3f}m | "
                  f"Z {data_mj.qpos[2]:.3f}m | "
                  f"{'CRASHED' if crashed else 'timeout'}")
            reset_env(data_mj, model_mj)
            episode_steps  = 0
            episode_reward = 0.0
            prev_dist      = np.linalg.norm(TARGET_POS - data_mj.qpos[:3])

        # PPO UPDATE
        if total_steps % COLLECT_STEPS == 0:
            loss = trainer.update(buffer)
            print(f"  >>> PPO update at step {total_steps} | loss {loss:.4f}")
            torch.save(brain.state_dict(), 'drone_brain.pth')
            print(f"  >>> Saved checkpoint to drone_brain.pth")

        # VISUALIZE
        viewer.sync()
        elapsed = time.time() - step_start
        if model_mj.opt.timestep - elapsed > 0:
            time.sleep(model_mj.opt.timestep - elapsed)