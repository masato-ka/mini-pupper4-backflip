"""Numbers from the spec (四足バックフリップ強化学習 実装仕様書). All provisional."""

import math
from dataclasses import dataclass

GRAVITY = 9.81

##
# Phases.
##

STAND, CROUCH, LAUNCH, AIR, LAND, DONE = range(6)
NUM_PHASES = 6
PHASE_NAMES = ("stand", "crouch", "launch", "air", "land", "done")

##
# Jump height command (Low / Mid / High).
##

LOW, MID, HIGH = range(3)
COMMAND_NAMES = ("low", "mid", "high")
# Observation scalar per command.
COMMAND_SCALARS = (0.0, 0.5, 1.0)
# Target jump height dh* [m].
JUMP_HEIGHTS = (0.25, 0.35, 0.45)
# Crouch depth ratio rho_c (T2 threshold h < rho_c * h_nom).
CROUCH_RATIOS = (0.80, 0.75, 0.70)


def takeoff_speed(dh: float) -> float:
  """v* = sqrt(2 g dh)."""
  return math.sqrt(2.0 * GRAVITY * dh)


def flight_time(dh: float) -> float:
  """t_f* = 2 v* / g."""
  return 2.0 * takeoff_speed(dh) / GRAVITY


def flip_rate(dh: float) -> float:
  """omega* = 2 pi / t_f*."""
  return 2.0 * math.pi / flight_time(dh)


##
# Termination / success levels L0 (loosest) .. L2 (final).
##


@dataclass(frozen=True)
class LevelCfg:
  # TM-6: fail if f_flip at touchdown is below this (None = disabled).
  min_f_flip_at_touchdown: float | None
  # TM-5: fail if theta > 2 pi + this.
  over_rotation_margin: float
  # TM-4: fail if theta < -this.
  forward_rotation_limit: float
  # TM-3: fail if base height < this [m].
  min_height: float
  # TM-7: multiplier on the per-phase time limits.
  phase_time_scale: float
  # Success (EV-9): theta >= 2 pi - this, held for success_hold_s.
  success_theta_margin: float
  success_hold_s: float


LEVELS = (
  LevelCfg(None, 2.0, math.pi, 0.08, 2.0, 1.0, 0.2),
  LevelCfg(0.25, 1.6, 3 * math.pi / 4, 0.10, 1.5, 0.75, 0.3),
  LevelCfg(0.5, 1.2, math.pi / 2, 0.12, 1.0, 0.5, 0.4),
)

# TM-7 per-phase time limits at L2 [s] (Stand has none).
PHASE_TIME_LIMITS_S = {CROUCH: 1.0, LAUNCH: 0.6, AIR: 1.2, LAND: 1.5}

# f_flip = clip(theta_td / (2 pi - F_FLIP_MARGIN), 0, 1).
F_FLIP_MARGIN = 0.5

##
# Stages S1..S4 (index 0..3): allowed commands and termination level.
##


@dataclass(frozen=True)
class StageCfg:
  commands: tuple[int, ...]
  level: int


STAGES = (
  StageCfg(commands=(MID,), level=0),
  StageCfg(commands=(MID,), level=0),
  StageCfg(commands=(LOW, MID, HIGH), level=1),
  StageCfg(commands=(LOW, MID, HIGH), level=2),
)

##
# Trigger and command input (EV-3 .. EV-8).
##

TRIGGER_DELAY_RANGE_S = (0.5, 1.5)  # EV-3
NO_TRIGGER_PROB = 0.15  # EV-4 (10-20 %)
COMMAND_CHANGE_PROB = 0.2  # EV-7
INPUT_JITTER_STEPS = 1  # EV-8: 0..1 step
# EV-6 interlock: stable for this long before the trigger is accepted.
# "Stable" is judged on low-pass filtered signals (first-order, time constant
# STABLE_FILTER_TAU_S): mean |joint velocity| over the 12 joints and the base
# angular speed. Instantaneous per-joint maxima never stayed below the
# thresholds under exploration noise (2nd Colab run, 2026-11-14: 0.5 % of
# episodes triggered); the filtered mean accepts 95 % within ~0.6 s.
STABLE_HOLD_S = 0.3
STABLE_FILTER_TAU_S = 0.1
STABLE_MAX_ANG_VEL = 0.5  # [rad/s], base, filtered
STABLE_MAX_JOINT_VEL = 1.0  # [rad/s], mean over joints, filtered

##
# Phase transition thresholds.
##

# theta uses the gravity angle only while |(g_x, g_z)| >= this (roll < 60 deg);
# otherwise it integrates the gyro pitch rate.
THETA_MIN_PROJECTION = 0.5
TAKEOFF_AIR_STEPS = 2  # T3: n_c == 0 for this many consecutive steps.
TAKEOFF_MIN_VZ_RATIO = 0.3  # T3: and v_z >= 0.3 v* (Mid: 0.79 m/s, ~3 cm rise).
TOUCHDOWN_MIN_AIR_STEPS = 2  # T4: steps in Air before a contact counts.
# T5 / EV-9 "upright" and "slow".
UPRIGHT_MIN_COS = math.cos(math.radians(25.0))
SETTLED_MAX_LIN_VEL = 0.3  # [m/s]
SETTLED_MAX_ANG_VEL = 1.0  # [rad/s]
SUCCESS_TAIL_S = 0.5  # EV-12 / TM-9: keep running after success.
TRIGGER_TIME_NORM_S = 2.0  # Observation: t_since_trigger / 2 s, clipped to 1.

##
# Domain randomization (reset-time). "full" from the spec; "narrow" is used in
# S1-S2 and widened from S3 by the curriculum (M5).
##

DR_FULL = {
  "encoder_bias": (-0.02, 0.02),  # [rad]
  "foot_friction": (0.2, 1.5),
  "added_mass": (-1.0, 3.0),  # [kg], base
  "com_offset": (-0.01, 0.01),  # [m], each axis
  "kp_scale": (0.8, 1.2),
  "kd_scale": (0.8, 1.2),
}
DR_NARROW = {
  "encoder_bias": (-0.01, 0.01),
  "foot_friction": (0.5, 1.0),
  "added_mass": (-0.5, 1.0),
  "com_offset": (-0.005, 0.005),
  "kp_scale": (0.9, 1.1),
  "kd_scale": (0.9, 1.1),
}

##
# Sensor model (initial values; Genesis-backflip noise as the anchor).
##

NOISE = {
  "gyro": 0.1,  # [rad/s]
  "acc": 0.2,  # [m/s^2] (not in the spec; provisional)
  "gravity": 0.02,
  "joint_pos": 0.01,  # [rad]
  "joint_vel": 0.5,  # [rad/s]
  "torque": 0.5,  # [Nm] (not in the spec; provisional)
}
GYRO_LIMIT = math.radians(2000.0)  # +-2000 dps
ACC_LIMIT = 16.0 * GRAVITY  # +-16 g
GYRO_BIAS = (-0.02, 0.02)  # [rad/s]
ACC_BIAS = (-0.2, 0.2)  # [m/s^2]
IMU_SCALE_ERROR = (-0.02, 0.02)
OBS_DELAY_STEPS = (1, 2)  # Policy steps.
OBS_HISTORY = 4
ACTION_DELAY_PHYSICS_STEPS = 4  # 0.02 s at dt = 0.005.
ACTION_SCALE = 0.5
EPISODE_LENGTH_S = 5.0

##
# Rewards. Functions return non-negative magnitudes; the sign is in the weight.
# Initial values from the spec (Genesis-backflip anchors); tune from the logs.
##

REWARD_WEIGHTS = {
  # Common.
  "action_rate": -0.002,
  "actions_symmetry": -0.1,
  "joint_limits": -1.0,
  "roll": -10.0,  # Launch..Land
  "yaw_rate": -1.0,  # Launch..Land
  "terminal_penalty": -5.0,  # Once per failure (sparse, not scaled by dt).
  # Survival bonus per second in Stand and Done only; not in the original spec
  # (added 2026-11-14, restricted to Stand/Done the same day: in the other
  # phases it paid for loitering in Crouch / Land).
  # Without it the Stand terms are all penalties and the first run (Colab,
  # 2026-11-14_05-49-16) learned to drop the torso within 0.3 s.
  "alive": 1.0,
  "leg_contact": -1.0,  # Not in the spec's table: thigh/calf contact is penalized, not terminal (TM-2 note).
  # Stand.
  "stand_height": -10.0,
  "stand_pose": -0.5,
  "stand_still": -1.0,
  "feet_distance": -1.0,
  # Crouch.
  "crouch_depth": 2.0,  # Sparse: once at T2.
  "crouch_level": -5.0,
  "feet_stay": -30.0,
  "no_early_pop": -2.0,
  # Launch.
  "takeoff_vz": 10.0,
  "vz_overshoot": -10.0,
  "pitch_kick": 3.0,
  "rear_before_front": -5.0,
  # Air.
  "apex_bonus": 10.0,  # Sparse.
  "apex_overshoot": -20.0,
  "flip_progress": 5.0,
  "wrong_direction": -0.5,
  "tuck": 0.0,  # Optional in the spec (-1); off by default.
  "land_prep": 0.0,  # Optional in the spec (+3); off by default.
  # Land (positive terms are multiplied by f_flip).
  "upright": 5.0,
  "land_height": 2.0,
  "feet_all_contact": 1.0,
  "land_damp": -1.0,
  "land_effort": -5e-5,
  "land_action_rate": -0.01,
  "success_bonus": 20.0,  # Sparse; ramped 0 -> 20 by the curriculum (S2).
}
# Optional-term weights from the spec, for enabling them from the CLI.
TUCK_WEIGHT_SPEC = -1.0
LAND_PREP_WEIGHT_SPEC = 3.0

# Kernel widths and thresholds.
CROUCH_DEPTH_STD = 0.03
FEET_STAY_HEIGHT = 0.02  # [m]
VZ_OVERSHOOT_RATIO = 1.1
APEX_STD = 0.06
APEX_OVERSHOOT_RATIO = 1.2
FLIP_PROGRESS_STD = 0.8
TUCK_THETA_RANGE = (0.25 * math.pi, 1.5 * math.pi)
LAND_PREP_THETA = 1.5 * math.pi
LAND_PREP_STD = 0.5
UPRIGHT_STD = 0.3
LAND_HEIGHT_STD = 0.05
# Tuck pose as offsets from the default pose (thigh forward, knee bent).
TUCK_OFFSETS = {"thigh": 0.6, "calf": -0.7}

##
# Curriculum (mdp/curriculum.py). Iterations are policy iterations.
##

CURRICULUM = {
  # rho per transition: S1->S2 rotation rate (not in the spec, provisional),
  # S2->S3 Mid success rate (spec L0->L1), S3->S4 min per-command success (L1->L2).
  "thresholds": (0.4, 0.4, 0.7),
  "promote_iters": 100,  # N
  "demote_iters": 200,
  "min_iters": 500,
  "success_ramp_iters": 1000,  # S2: success_bonus 0 -> 20.
  "num_steps_per_env": 24,  # Must match the runner.
  "ema_episodes": 2000,
  "start_stage": 0,
  # >= 0 pins the stage (debugging); -1 = automatic. An int (not None) so the
  # train CLI can set it: tyro infers the flag type from this default.
  "fixed_stage": -1,
  # EV-6 interlock is enforced from this stage on (3 = S4); before it the
  # trigger is accepted at its input time. See curriculum.py.
  "interlock_from_stage": 3,
}
