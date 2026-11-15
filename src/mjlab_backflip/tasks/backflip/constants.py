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

# EV-3. Spec: U[0.5, 1.5]; widened to 2.5 s on 2026-11-15: the mean policy of
# run 2026-11-15_04-43-40 started swaying after ~1-2 s of waiting, and with
# the interlock the real robot waits until it is still.
TRIGGER_DELAY_RANGE_S = (0.5, 2.5)
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
# Untriggered episodes run this long; triggered ones end at FLIP_TAIL_S.
# Latest trigger (2.5 s) + FLIP_TAIL_S fits.
EPISODE_LENGTH_S = 4.5

##
# Rewards (revised 2026-11-15, Genesis-backflip style).
#
# The flip terms are gated by the time since the accepted trigger, not by the
# phase state machine: four Colab runs with phase-gated rewards each found a
# loophole in the gating (dropping the torso, avoiding the trigger, farming
# Crouch, faking the takeoff). The phases are still tracked for the success
# judgement, the metrics and the curriculum.
#
# Weights and windows follow ziyanx02/Genesis-backflip (reward_wrapper.py,
# train_backflip.py), whose Go2 learns a backflip in ~1000 iterations. Its
# go signal is episode time 0.5 s; here it is the trigger plus FLIP_START_S.
# Functions return the raw value; the weight carries the sign. flip_ang_vel
# and flip_lin_vel_z are signed (rotating / moving the wrong way costs).
##

REWARD_WEIGHTS = {
  # Flip (time windows below).
  "flip_orientation": -1.0,  # ||g_b - g_ref(tau)||^2, all the time.
  "flip_ang_vel": 5.0,  # clip(omega_back, +-7.2), rotation window.
  "flip_lin_vel_z": 20.0,  # min(v_z, 3), lift window.
  "feet_height_before": -30.0,  # sum relu(z_foot - 0.02), before the flip.
  "height_control": -10.0,  # (h - h_nom)^2, before the trigger and after the flip.
  # Not in Genesis (added 2026-11-15): standing after one full turn. Without
  # it the landing was almost free (Colab run 2026-11-15_00-56-47: the mean
  # policy over-rotated to 1.5 turns and landed on its back).
  "flip_landing": 20.0,  # kern(theta - 2 pi) kern(h - h_nom) n_c / 4, after the rotation.
  # Always on.
  "gravity_y": -10.0,  # g_y^2
  "ang_vel_z": -1.0,  # |omega_z|
  "actions_symmetry": -0.1,
  "feet_distance": -1.0,
  "action_rate": -0.001,
  "joint_limits": -1.0,  # Not in Genesis; kept from the spec for the real robot.
  # Not in Genesis (added 2026-11-15): stand still until the trigger, so the
  # EV-6 interlock accepts it on the real robot (the interlock is off in
  # training; the mean policy of model_750 passed it in 0-2 % of episodes).
  # -0.1 -> -0.5 (2026-11-15): at -0.1 the mean policy's swaying (raw ~3)
  # was lost under the exploration noise (raw ~5.6 at std 0.75) and cost
  # only ~0.3 /s; model_3000 swayed with the Low command (3 % accepted).
  "stand_still": -0.5,  # EV-6 filtered |omega| + mean |q_dot|, before the trigger.
}

# Time windows, tau = time since the accepted trigger [s].
FLIP_START_S = 0.1  # Feet held down before this; height control off from tau = 0.
FLIP_LIFT_S = 0.25  # flip_lin_vel_z in [start, start + this).
FLIP_ROTATE_S = 0.5  # flip_ang_vel in [start, start + this); g_ref turns 2 pi over it.
FLIP_SETTLE_S = 0.9  # height_control again from start + this.
# Truncate the episode at tau = this (time out). Was 1.5 (Genesis); the
# policy touched down at tau ~ 1.0 s, leaving too little time to settle.
FLIP_TAIL_S = 2.0
LIN_VEL_Z_CLIP = 3.0  # [m/s]
ANG_VEL_CLIP = 7.2  # [rad/s]
FEET_LIFT_TOLERANCE = 0.02  # [m]
LANDING_THETA_STD = 0.5  # [rad]
# "Landed" (curriculum metric, 2026-11-15): in the last LANDED_CHECK_S
# before the flip tail, |theta - 2 pi| <= the level's success margin, upright,
# h > LANDED_MIN_HEIGHT and at least LANDED_MIN_FEET feet down. Unlike EV-9 it
# has no stillness / hold condition, which training's exploration noise
# never lets pass (std 0.75: EV-9 0 %, landed 73 %; mean action: 64 % / 78 %).
LANDED_CHECK_S = 0.1
LANDED_MIN_HEIGHT = 0.22  # [m]
LANDED_MIN_FEET = 3
LANDING_HEIGHT_STD = 0.1  # [m]

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
  "num_steps_per_env": 24,  # Must match the runner.
  "ema_episodes": 2000,
  "start_stage": 0,
  # >= 0 pins the stage (debugging); -1 = automatic. An int (not None) so the
  # train CLI can set it: tyro infers the flag type from this default.
  "fixed_stage": -1,
  # EV-6 interlock is enforced from this stage on; -1 = never (training
  # accepts the trigger at its input time). See curriculum.py.
  "interlock_from_stage": -1,
}
