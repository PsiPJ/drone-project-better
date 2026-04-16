import mujoco
import numpy as np
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/bitcraze_crazyflie_2/scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        quat = data.sensor("body_quat").data
        ang_vel = data.sensor("body_gyro").data

        #-- Translational States (Where is it in the real World) -- 
        current_position = data.qpos[:3] # [x,y,z] 
        current_velocity = data.qvel[:3] # [vx,vy,vz] 

        #-- Rotational States (How is it Tilted? = Attitude) -- 
        attitude_quat = data.sensor("body_quat").data # [w,x,y,z] 
        angular_velocity = data.sensor("body_gyro").data # [wx,wy,wz] 

        #-- Position Control (Where do we want to be?) -- HOW 

        # Current Pos and Vel 
        x_pos, y_pos, z_pos = current_position
        x_vel, y_vel, z_vel = current_velocity

        # USER DEFINED TARGETS
        target_x, target_y, target_z = 0.8, 0, 1.0 #Hover at 1m 
        target_yaw = 0.0 # No Rotation 
        goal_yaw = target_yaw

        # Constants (Gains)
        kp_z, kd_z = 8.0, 4.0          # Keep height strong and stable
        kp_xy, kd_xy = 0.5, 1.2        # A patient brain: gentle acceleration and strong brakes
        kp_angle, kd_angle = 5.0, 0.4  # Lightning-fast muscles: snap to angles instantly

        # --- HEIGHT CONTROL (thrust) ---
        # Goal: Adjust thrust to reach target_z and stop vertical drifting (z_vel)
        
        HOVER_THRUST = 0.26487 # Constant needed to fight Gravity 
        thrust_action = HOVER_THRUST + kp_z * (target_z - z_pos) - kd_z * z_vel
        thrust_action = max(0.0, min(thrust_action, 1.0))  # Clamp thrust

        # --- ATTITUDE CONTROL (Stabilization) ---
        # Quaternion to Roll, Pitch, Yaw conversion 
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, attitude_quat)

        # rotation matrix → Euler (ZYX)
        R = mat.reshape(3,3)

        current_roll  = np.arctan2(R[2,1], R[2,2])
        current_pitch = -np.arcsin(R[2,0])
        current_yaw   = np.arctan2(R[1,0], R[0,0])

        # --- TRANSLATIONAL CONTROL (x,y) - tilt ---
        # Goal: Decide how much to lean to get to the target (x,y).
        # Note: In MuJoCo's coordinate system for this drone:
        # To move +X (Forward), we need a negative Pitch.
        # To move +Y (Left), we need a positive Roll.

        MAX_TILT_ANGLE = 0.12

        # Position error in world frame
        x_error = target_x - x_pos
        y_error = target_y - y_pos

        # Rotate error into body frame
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)

        x_body_error =  cos_yaw * x_error + sin_yaw * y_error
        y_body_error = -sin_yaw * x_error + cos_yaw * y_error

        # Rotate velocity into body frame (FIXED)
        x_body_vel =  cos_yaw * x_vel + sin_yaw * y_vel
        y_body_vel = -sin_yaw * x_vel + cos_yaw * y_vel

        # Add a velocity target of zero. Instead of just penalizing current velocity, explicitly demand the drone be stationary at the target::
        raw_pitch = -(kp_xy * x_body_error + kd_xy * x_body_vel)
        raw_roll  =  (kp_xy * y_body_error + kd_xy * y_body_vel)

        goal_pitch = max(min(raw_pitch, MAX_TILT_ANGLE), -MAX_TILT_ANGLE)
        goal_roll  = max(min(raw_roll, MAX_TILT_ANGLE), -MAX_TILT_ANGLE)

        # Calculate Angle Errors
        roll_error  = goal_roll  - current_roll
        pitch_error = goal_pitch - current_pitch
        yaw_error   = goal_yaw   - current_yaw   # FIXED

        roll_torque_action  = -kp_angle * roll_error  - kd_angle * angular_velocity[0]
        pitch_torque_action = -kp_angle * pitch_error - kd_angle * angular_velocity[1]
        yaw_torque_action   = -kp_angle * yaw_error   - kd_angle * angular_velocity[2]

        # EXECUTE THE ACTIONS  
        data.ctrl[:] = [thrust_action, roll_torque_action, pitch_torque_action, yaw_torque_action]
        

        mujoco.mj_step(model, data)
        viewer.sync()

        # Real-time synchronization
        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))