import mujoco
import mujoco.viewer
import time
import numpy as np

HOVER_THRUST = 0.26487

kp_angle = 5.0
kd_angle = 0.15

kp_z = 2.0
kd_z = 1.0
target_z = 0.1

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/bitcraze_crazyflie_2/scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # --- Sensors ---
        quat = data.sensor("body_quat").data
        ang_vel = data.sensor("body_gyro").data

        z = data.qpos[2]
        vz = data.qvel[2]

        # --- Height control ---
        thrust = HOVER_THRUST + kp_z * (target_z - z) - kd_z * vz

        # --- Orientation control (small-angle approx) ---
        roll_error  = quat[1]
        pitch_error = quat[2]
        yaw_error   = quat[3]

        roll_torque  = -kp_angle * roll_error  - kd_angle * ang_vel[0]
        pitch_torque = -kp_angle * pitch_error - kd_angle * ang_vel[1]
        yaw_torque   = -kp_angle * yaw_error   - kd_angle * ang_vel[2]

        # --- Apply control ---
        data.ctrl[:] = [thrust, roll_torque, pitch_torque, yaw_torque]

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
