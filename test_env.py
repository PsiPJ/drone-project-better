# ============================================================
# 🧪 SIMPLE ENVIRONMENT TEST LOOP
# ============================================================
# This is NOT training.
# This is just to verify:
# - MuJoCo runs
# - environment resets properly
# - step() works
# - rewards are being computed

import mujoco.viewer
import time
import numpy as np
from fire_env import FireDroneEnv


# ============================================================
# 🔧 CREATE ENVIRONMENT
# ============================================================
# IMPORTANT: point to your scene.xml file

env = FireDroneEnv("mujoco_menagerie/bitcraze_crazyflie_2/scene.xml")

obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    for step in range(1000):

        action = np.random.uniform(-1, 1, size=(4,)).astype(np.float32)

        obs, reward, done, truncated, info = env.step(action)

        print(f"Step {step} | Reward: {reward:.3f}")

        viewer.sync()

        if done:
            obs, info = env.reset()

        time.sleep(0.01)