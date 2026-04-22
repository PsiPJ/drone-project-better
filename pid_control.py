"""
PID control for the MuJoCo Crazyflie 2 scene (scene.xml + cf2.xml).

Extends pd_control.py with integral terms on altitude, XY position, and
attitude to eliminate steady-state error from gravity imbalance, aero drag,
or CoG offsets.

Anti-windup: each integrator is clamped to a symmetric bound every step,
which prevents unbounded accumulation when the actuator is saturated.

Sign convention (same as pd_control.py):
  - Moment gear is negative (-1e-5), so ctrl must be negative to produce
    a positive (restoring) torque. The kp term is negated; ki follows suit.

Run from repo root:
    python pid_control.py
    python pid_control.py --viewer
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from pd_control import GRAVITY, HOVER_THRUST, Setpoint, quat_wxyz_to_rpy_zyx

_SCENE = (
    Path(__file__).resolve().parent
    / "mujoco_menagerie"
    / "bitcraze_crazyflie_2"
    / "scene.xml"
)


@dataclass
class PIDGains:
    # Altitude
    kp_z: float = 7.5
    kd_z: float = 4.5
    ki_z: float = 2.0

    # Horizontal position (world-frame, cascaded into attitude)
    kp_xy: float = 0.3
    kd_xy: float = 1.1
    ki_xy: float = 0.05

    # Attitude (roll / pitch)
    kp_angle: float = 8.0
    kd_angle: float = 8.8   # positive due to negative moment gear
    ki_angle: float = 0.5
    kd_yaw: float = 2.0     # yaw P is fragile near ±π; damp rate only

    # Tilt limits and filter
    max_tilt: float = 0.15
    tilt_filter_alpha: float = 0.12

    # Anti-windup clamp bounds (symmetric)
    max_int_z: float = 0.15    # thrust units
    max_int_xy: float = 0.3    # m/s (position-error integral)
    max_int_angle: float = 0.08  # moment ctrl units


@dataclass
class PIDState:
    """Persistent filter + integrator state. Reset whenever the simulation resets."""

    # Tilt low-pass (same role as PDState)
    pitch_filt: float = 0.0
    roll_filt: float = 0.0

    # Integrators
    int_z: float = 0.0
    int_x: float = 0.0
    int_y: float = 0.0
    int_roll: float = 0.0
    int_pitch: float = 0.0


def compute_pid_control(
    data: mujoco.MjData,
    sp: Setpoint,
    gains: PIDGains,
    dt: float,
    hover_thrust: float = HOVER_THRUST,
    state: PIDState | None = None,
) -> np.ndarray:
    """
    Compute ctrl[4] = [thrust, roll_moment, pitch_moment, yaw_moment].

    Args:
        data:          MuJoCo data (after mj_forward / mj_step).
        sp:            Position + yaw setpoint.
        gains:         PID gain struct.
        dt:            Simulation timestep (model.opt.timestep) for integration.
        hover_thrust:  Feed-forward thrust at hover.
        state:         Persistent PIDState; if None a throw-away state is used
                       (integrators have no effect across calls).
    """
    st = state if state is not None else PIDState()

    pos = data.qpos[:3]
    vel = data.qvel[:3]
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])

    quat = data.sensor("body_quat").data
    omega = data.sensor("body_gyro").data
    roll, pitch, _yaw = quat_wxyz_to_rpy_zyx(quat)

    # ── Altitude PID ──────────────────────────────────────────────────────────
    ez = sp.z - z
    st.int_z = float(np.clip(st.int_z + ez * dt, -gains.max_int_z, gains.max_int_z))
    thrust = hover_thrust + gains.kp_z * ez - gains.kd_z * vz + gains.ki_z * st.int_z
    thrust = float(np.clip(thrust, 0.0, 1.0))

    # ── Horizontal position PID → body tilt setpoint ──────────────────────────
    ex = sp.x - x
    ey = sp.y - y
    st.int_x = float(np.clip(st.int_x + ex * dt, -gains.max_int_xy, gains.max_int_xy))
    st.int_y = float(np.clip(st.int_y + ey * dt, -gains.max_int_xy, gains.max_int_xy))

    ax_w = gains.kp_xy * ex - gains.kd_xy * vx + gains.ki_xy * st.int_x
    ay_w = gains.kp_xy * ey - gains.kd_xy * vy + gains.ki_xy * st.int_y

    # Rotate world acceleration into body frame using setpoint yaw (avoids drift coupling)
    c, s = np.cos(sp.yaw), np.sin(sp.yaw)
    ax_b =  c * ax_w + s * ay_w
    ay_b = -s * ax_w + c * ay_w
    mt = gains.max_tilt
    gp_raw = float(np.clip(ax_b / GRAVITY, -mt, mt))
    gr_raw = float(np.clip(-ay_b / GRAVITY, -mt, mt))

    alpha = gains.tilt_filter_alpha
    if alpha > 0.0:
        st.pitch_filt = (1.0 - alpha) * st.pitch_filt + alpha * gp_raw
        st.roll_filt  = (1.0 - alpha) * st.roll_filt  + alpha * gr_raw
        goal_pitch, goal_roll = st.pitch_filt, st.roll_filt
    else:
        goal_pitch, goal_roll = gp_raw, gr_raw

    # ── Attitude PID ──────────────────────────────────────────────────────────
    # Moment gear is -1e-5, so the P term is negated to get restoring torque.
    # The I term follows the same sign: -ki * int accumulates more negative ctrl
    # when the error has been persistently positive, producing more positive torque.
    e_roll  = goal_roll  - roll
    e_pitch = goal_pitch - pitch
    st.int_roll  = float(np.clip(st.int_roll  + e_roll  * dt, -gains.max_int_angle, gains.max_int_angle))
    st.int_pitch = float(np.clip(st.int_pitch + e_pitch * dt, -gains.max_int_angle, gains.max_int_angle))

    u_roll  = -gains.kp_angle * e_roll  + gains.kd_angle * float(omega[0]) - gains.ki_angle * st.int_roll
    u_pitch = -gains.kp_angle * e_pitch + gains.kd_angle * float(omega[1]) - gains.ki_angle * st.int_pitch
    u_yaw   = +gains.kd_yaw * float(omega[2])

    return np.array([thrust, u_roll, u_pitch, u_yaw], dtype=np.float64)


# ── Simulation runners ────────────────────────────────────────────────────────

def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    steps: int,
    sp: Setpoint,
    gains: PIDGains,
    state: PIDState,
) -> None:
    dt = model.opt.timestep
    for _ in range(steps):
        data.ctrl[:] = compute_pid_control(data, sp, gains, dt, state=state)
        mujoco.mj_step(model, data)
    p = data.qpos[:3]
    print(f"After {steps} steps: pos [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]  "
          f"target [{sp.x:.3f}, {sp.y:.3f}, {sp.z:.3f}]")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sp: Setpoint,
    gains: PIDGains,
    state: PIDState,
) -> None:
    import mujoco.viewer

    dt = model.opt.timestep
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            data.ctrl[:] = compute_pid_control(data, sp, gains, dt, state=state)
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)


def main() -> None:
    p = argparse.ArgumentParser(description="PID controller for Crazyflie MuJoCo scene")
    p.add_argument("--scene", type=Path, default=_SCENE)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--x",   type=float, default=0.8)
    p.add_argument("--y",   type=float, default=0.0)
    p.add_argument("--z",   type=float, default=1.0)
    p.add_argument("--yaw", type=float, default=0.0)
    args = p.parse_args()

    if not args.scene.is_file():
        raise SystemExit(f"Scene not found: {args.scene}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    sp    = Setpoint(x=args.x, y=args.y, z=args.z, yaw=args.yaw)
    gains = PIDGains()
    state = PIDState()

    if args.viewer:
        run_viewer(model, data, sp, gains, state)
    else:
        run_headless(model, data, args.steps, sp, gains, state)


if __name__ == "__main__":
    main()
