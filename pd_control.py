"""
PD control for the MuJoCo Crazyflie 2 scene (scene.xml + cf2.xml).

The model has four motors: thrust along body +Z, and roll/pitch/yaw moments (small gear
scaling in XML). Sensors: framequat at IMU, gyro at IMU.

Control structure (cascade):
  1) Horizontal: world-frame acceleration PD (a_w = kp*e - kd*v_w), rotate by *setpoint*
     yaw into body tilt / g. Setpoint yaw avoids drift coupling; optional tilt low-pass.
  2) Altitude: PD on z and vz + hover feedforward -> thrust ctrl in [0, 1] (clamped).
  3) Attitude: PD on roll/pitch/yaw errors vs gyro -> moment ctrls in [-1, 1].

Run from repo root:  python pd_control.py
With viewer:         python pd_control.py --viewer
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


# Default scene path: same folder layout as this repo (mujoco_menagerie/.../scene.xml)
_SCENE = Path(__file__).resolve().parent / "mujoco_menagerie" / "bitcraze_crazyflie_2" / "scene.xml"
HOVER_THRUST = 0.26487  # matches cf2.xml keyframe hover ctrl
GRAVITY = 9.81




@dataclass
class PDGains:
    kp_z: float = 7.5
    kd_z: float = 4.5
    kp_xy: float = 0.3
    kd_xy: float = 1.1
    kp_angle: float = 8.0
    # Attitude kd must be POSITIVE because the moment gear is negative (-1e-5).
    # Sign derivation: tau = ctrl * (-1e-5); for damping, ctrl = +kd * omega_body.
    # Critical-damping value: 2 * sqrt(kp_angle * Iyy / gear) ≈ 8.8 for kp=8, Iyy=2.4e-5.
    kd_angle: float = 8.8
    kd_yaw: float = 2.0  # yaw-rate damping (Euler yaw P is fragile near ±π in this model)
    max_tilt: float = 0.15
    tilt_filter_alpha: float = 0.12


@dataclass
class PDState:
    """Mutable filter state; reset when you reset the simulation."""

    pitch_filt: float = 0.0
    roll_filt: float = 0.0


@dataclass
class Setpoint:
    x: float = 1.0
    y: float = 2.0
    z: float = 1.0
    yaw: float = 0.0


def quat_wxyz_to_rpy_zyx(q: np.ndarray) -> tuple[float, float, float]:
    """Body orientation from world: ZYX Euler (roll, pitch, yaw). q is MuJoCo [w,x,y,z]."""
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, q.astype(np.float64))
    r = mat.reshape(3, 3)
    roll = float(np.arctan2(r[2, 1], r[2, 2]))
    pitch = float(-np.arcsin(r[2, 0]))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    return roll, pitch, yaw


def wrap_angle_pi(e: float) -> float:
    return float(np.arctan2(np.sin(e), np.cos(e)))


def compute_pd_control(
    data: mujoco.MjData,
    sp: Setpoint,
    gains: PDGains,
    hover_thrust: float = HOVER_THRUST,
    state: PDState | None = None,
) -> np.ndarray:
    """
    Compute ctrl[4] = [thrust, roll_moment, pitch_moment, yaw_moment] for one timestep.
    Reads qpos/qvel and sensors from `data` (call after mj_forward/mj_step as usual).

    Pass a persistent :class:`PDState` to enable tilt low-pass filtering (reduces overshoot).
    """
    pos = data.qpos[:3]
    vel = data.qvel[:3]
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])

    quat = data.sensor("body_quat").data
    omega = data.sensor("body_gyro").data

    roll, pitch, yaw = quat_wxyz_to_rpy_zyx(quat)

    # Altitude PD
    thrust = hover_thrust + gains.kp_z * (sp.z - z) - gains.kd_z * vz
    thrust = float(np.clip(thrust, 0.0, 1.0))

    # World XY acceleration PD (consistent on e and v), then body tilt via setpoint yaw
    ex = sp.x - x
    ey = sp.y - y
    ax_w = gains.kp_xy * ex - gains.kd_xy * vx
    ay_w = gains.kp_xy * ey - gains.kd_xy * vy
    c, s = np.cos(sp.yaw), np.sin(sp.yaw)
    ax_b = c * ax_w + s * ay_w
    ay_b = -s * ax_w + c * ay_w
    mt = gains.max_tilt
    gp_raw = float(np.clip(ax_b / GRAVITY, -mt, mt))
    gr_raw = float(np.clip(-ay_b / GRAVITY, -mt, mt))

    alpha = gains.tilt_filter_alpha
    if state is not None and alpha > 0.0:
        a = alpha
        state.pitch_filt = (1.0 - a) * state.pitch_filt + a * gp_raw
        state.roll_filt = (1.0 - a) * state.roll_filt + a * gr_raw
        goal_pitch = state.pitch_filt
        goal_roll = state.roll_filt
    else:
        goal_pitch = gp_raw
        goal_roll = gr_raw

    # Attitude PD (body rates from gyro).
    # Note: moment gear is -1e-5, so ctrl = -kp*angle_err + kd*omega produces the correct
    # restoring torque direction.  Using -kd*omega would make the derivative term actively
    # divergent instead of damping.
    e_roll = goal_roll - roll
    e_pitch = goal_pitch - pitch
    # Yaw moment is tiny in this MJCF; Euler yaw can jump near ±π. Damping body yaw rate only.
    u_roll  = -gains.kp_angle * e_roll  + gains.kd_angle * float(omega[0])
    u_pitch = -gains.kp_angle * e_pitch + gains.kd_angle * float(omega[1])
    u_yaw   = +gains.kd_yaw * float(omega[2])

    return np.array([thrust, u_roll, u_pitch, u_yaw], dtype=np.float64)


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    steps: int,
    sp: Setpoint,
    gains: PDGains,
    state: PDState | None = None,
) -> None:
    st = state if state is not None else PDState()
    for _ in range(steps):
        data.ctrl[:] = compute_pd_control(data, sp, gains, state=st)
        mujoco.mj_step(model, data)
    p = data.qpos[:3]
    print(f"After {steps} steps: pos [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sp: Setpoint,
    gains: PDGains,
    state: PDState | None = None,
) -> None:
    import mujoco.viewer

    st = state if state is not None else PDState()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            data.ctrl[:] = compute_pd_control(data, sp, gains, state=st)
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


def main() -> None:
    p = argparse.ArgumentParser(description="PD controller for Crazyflie MuJoCo scene")
    p.add_argument(
        "--scene",
        type=Path,
        default=_SCENE,
        help="Path to scene.xml",
    )
    p.add_argument("--viewer", action="store_true", help="Open passive viewer")
    p.add_argument("--steps", type=int, default=15000, help="Headless simulation steps")
    p.add_argument("--x", type=float, default=0.8)
    p.add_argument("--y", type=float, default=0.0)
    p.add_argument("--z", type=float, default=1.0)
    p.add_argument("--yaw", type=float, default=0.0)
    args = p.parse_args()

    if not args.scene.is_file():
        raise SystemExit(f"Scene not found: {args.scene}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)  # populate sensors before first control step

    sp = Setpoint(x=args.x, y=args.y, z=args.z, yaw=args.yaw)
    gains = PDGains()
    pd_state = PDState()

    if args.viewer:
        run_viewer(model, data, sp, gains, state=pd_state)
    else:
        run_headless(model, data, args.steps, sp, gains, state=pd_state)


if __name__ == "__main__":
    main()
