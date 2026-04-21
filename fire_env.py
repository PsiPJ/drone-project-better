import gymnasium as gym
import numpy as np
import mujoco

from fire_logic import Fire


# ============================================================
# 🧠 CUSTOM MUJOCO RL ENVIRONMENT
# ============================================================
# This wraps your simulator so SAC can interact with it.

class FireDroneEnv(gym.Env):

    def __init__(self, model_path):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # ============================================================
# 🧠 DEBUG MODE SWITCH
# ============================================================
# Turn this on/off to control verbosity

        self.debug = True
        self.debug_every = 10  # print every N steps
        self.step_count = 0
        # ====================================================
        # 🔥 DEFINE FIRES IN THE WORLD
        # ====================================================
        self.fires = [
            Fire([0.5, 0.0, 0.01]),
            Fire([-0.4, 0.3, 0.01])
        ]
        # ============================================================
        # 🎨 VISUAL DEBUG STATE (for future MuJoCo coloring)
        # ============================================================
        self.fire_visual_intensity = [1.0 for _ in self.fires]
        # ====================================================
        # 🎯 DRONE START POSITION
        # ====================================================
        self.start_pos = np.array([0, 0, 0.1])

        # ====================================================
        # 🎮 ACTION SPACE (what agent controls)
        # ====================================================
        # [thrust, roll, pitch, yaw]
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        # ====================================================
        # 👁 OBSERVATION SPACE (what agent sees)
        # ====================================================
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Reset fire states
        for fire in self.fires:
            fire.intensity = 1.0

        # Return initial observation
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        # ====================================================
        # ⚙️ APPLY ACTION TO SIMULATION
        # ====================================================
        self.data.ctrl[:] = action

        # Step physics forward
        mujoco.mj_step(self.model, self.data)

        # ====================================================
        # 📡 GET DRONE STATE
        # ====================================================
        drone_pos = self.data.qpos[:3]
        drone_vel = self.data.qvel[:3]

        # ====================================================
        # 🔥 UPDATE FIRES
        # ====================================================
        for fire in self.fires:
            fire.update(drone_pos)
            # ============================================================
        # 🔥 DEBUG: DETECT IF FIRE IS ACTUALLY CHANGING
        # ============================================================

        for i, fire in enumerate(self.fires):

            # If fire is actively decreasing, print a signal
            if fire.intensity < 1.0 and fire.intensity > 0.0:
                print(f"🔥 FIRE {i} BURNING DOWN → {fire.intensity:.3f}")

            # If fire is extinguished
            if fire.is_out():
                print(f"💨 FIRE {i} EXTINGUISHED")
                # ============================================================
        # 📡 DRONE POSITION DEBUG RELATIVE TO FIRE
        # ============================================================

        for i, fire in enumerate(self.fires):
            dist = np.linalg.norm(drone_pos - fire.position)

            print(f"   → Fire {i} distance: {dist:.3f}")
                # ============================================================
                # 🎨 SYNC VISUAL STATE WITH LOGIC STATE
                # ============================================================
        for i, fire in enumerate(self.fires):
            self.fire_visual_intensity[i] = fire.intensity

# ============================================================
# 🔥 DEBUG: FIRE STATUS + DISTANCE (COMBINED LINE)
# ============================================================
# This prints BOTH:
# - fire intensity (how alive it is)
# - distance from drone (interaction signal)
# in one compact readable line.

        if self.debug and self.step_count % self.debug_every == 0:

            status = " | ".join(
                f"F{i}: I={fire.intensity:.2f} D={np.linalg.norm(drone_pos - fire.position):.3f}"
                for i, fire in enumerate(self.fires)
            )

            print(f"[STEP {self.step_count}] FIRE STATUS → {status}")

        # ====================================================
        # 🎯 REWARD FUNCTION
        # ====================================================
        reward = 0.0

        for fire in self.fires:
            dist = np.linalg.norm(drone_pos - fire.position)

            # Reward being close to fire
            reward += 1.0 / (1.0 + dist)

            # Big reward if fire is extinguished
            if fire.is_out():
                reward += 50.0

        # Crash penalty (simple version)
        crashed = drone_pos[2] < 0.05
        if crashed:
            reward -= 100.0

        # ====================================================
        # 🧠 DONE CONDITION
        # ====================================================
        done = all(fire.is_out() for fire in self.fires)

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        drone_pos = self.data.qpos[:3]
        drone_vel = self.data.qvel[:3]

        quat = self.data.sensor("body_quat").data
        gyro = self.data.sensor("body_gyro").data

        fire_obs = []

        for fire in self.fires:
            fire_obs.extend(fire.position - drone_pos)
            fire_obs.append(fire.intensity)

        return np.concatenate([
            drone_pos,
            drone_vel,
            quat,
            gyro,
            np.array(fire_obs)
        ]).astype(np.float32)
        
        