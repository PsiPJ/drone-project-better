import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/bitcraze_crazyflie_2/scene.xml")
data = mujoco.MjData(model)

# Target point B
target_x, target_y, target_z = 0.5, 0.2, 1.0
target_yaw = 0.0

kp_z, kd_z = 8.0, 4.0
kp_xy, kd_xy = 0.5, 1.2
kp_angle, kd_angle = 15000.0, 1500.0
MAX_TILT_ANGLE = 0.12
HOVER_THRUST = 0.26487

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        #Get Current State from MuJoCo Sensors
        current_position = data.qpos[:3]
        current_velocity = data.qvel[:3]
        attitude_quat = data.sensor("body_quat").data #Gets the current attitude in quaternion form [w,x,y,z]
        angular_velocity = data.sensor("body_gyro").data #Gets the current angular velocity in body frame [wx,wy,wz]

        x_pos, y_pos, z_pos = current_position
        x_vel, y_vel, z_vel = current_velocity

        # current position
        print(f"Current position: x={x_pos:.3f}, y={y_pos:.3f}, z={z_pos:.3f}")

        # position errors
        x_error = target_x - x_pos
        y_error = target_y - y_pos
        z_error = target_z - z_pos

        # height PD control
        thrust = HOVER_THRUST + kp_z * z_error - kd_z * z_vel
        thrust = float(np.clip(thrust, 0.0, 1.0))

        # attitude from quaternion
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, attitude_quat)
        R = mat.reshape(3, 3)

        current_roll = np.arctan2(R[2, 1], R[2, 2])
        current_pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
        current_yaw = np.arctan2(R[1, 0], R[0, 0])

        # rotate world error into body frame
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)

        x_body_error = cos_yaw * x_error + sin_yaw * y_error
        y_body_error = -sin_yaw * x_error + cos_yaw * y_error

        x_body_vel = cos_yaw * x_vel + sin_yaw * y_vel
        y_body_vel = -sin_yaw * x_vel + cos_yaw * y_vel

        raw_pitch = kp_xy * x_body_error - kd_xy * x_body_vel
        raw_roll = -(kp_xy * y_body_error - kd_xy * y_body_vel)

        goal_pitch = float(np.clip(raw_pitch, -MAX_TILT_ANGLE, MAX_TILT_ANGLE))
        goal_roll = float(np.clip(raw_roll, -MAX_TILT_ANGLE, MAX_TILT_ANGLE))

        roll_error = goal_roll - current_roll
        pitch_error = goal_pitch - current_pitch
        yaw_error = target_yaw - current_yaw

        roll_torque = kp_angle * roll_error - kd_angle * angular_velocity[0]
        pitch_torque = kp_angle * pitch_error - kd_angle * angular_velocity[1]
        yaw_torque = kp_angle * yaw_error - kd_angle * angular_velocity[2]

        data.ctrl[:] = [thrust, roll_torque, pitch_torque, yaw_torque]

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))