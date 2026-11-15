"""Backflip environment configuration (Unitree Go2 stand-in).

Built from scratch rather than by stripping the velocity task: only the Go2
asset, the flat terrain and the foot contact sensor are shared with
``tasks/velocity/go2_env_cfg.py``.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_backflip.robots import GO2_SPEC, get_go2_robot_cfg
from mjlab_backflip.tasks.backflip import constants as C
from mjlab_backflip.tasks.backflip import mdp
from mjlab_backflip.tasks.velocity.go2_env_cfg import (
  FOOT_GEOM_NAMES,
  FOOT_NAMES,
  feet_ground_contact_cfg,
)

FOOT_RADIUS = 0.022  # Foot collision sphere in go2.xml.


def _robot_cfg():
  """Go2 with the fixed 0.02 s action (actuator command) delay."""
  robot = get_go2_robot_cfg()
  assert robot.articulation is not None
  robot.articulation = replace(
    robot.articulation,
    actuators=tuple(
      replace(
        a,
        delay_min_lag=C.ACTION_DELAY_PHYSICS_STEPS,
        delay_max_lag=C.ACTION_DELAY_PHYSICS_STEPS,
      )
      for a in robot.articulation.actuators
    ),
  )
  return robot


def _sensors():
  base_ground = ContactSensorCfg(
    name="base_ground_contact",
    primary=ContactMatch(mode="geom", pattern=GO2_SPEC.base_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  return (feet_ground_contact_cfg(), base_ground)


def _history(term: ObservationTermCfg, delay: bool = True) -> ObservationTermCfg:
  term.history_length = C.OBS_HISTORY
  if delay:
    term.delay_min_lag, term.delay_max_lag = C.OBS_DELAY_STEPS
  return term


def _observations() -> dict[str, ObservationGroupCfg]:
  n = C.NOISE
  # Per-step actor terms, stacked over 4 steps: 6 terms, 54 values per step.
  actor_history = {
    "gyro": _history(
      ObservationTermCfg(
        func=mdp.imu_reading,
        params={
          "sensor_name": "robot/imu_ang_vel",
          "limit": C.GYRO_LIMIT,
          "bias_range": C.GYRO_BIAS,
          "scale_range": C.IMU_SCALE_ERROR,
        },
        noise=Unoise(n_min=-n["gyro"], n_max=n["gyro"]),
      )
    ),
    "acc": _history(
      ObservationTermCfg(
        func=mdp.imu_reading,
        params={
          "sensor_name": "robot/imu_lin_acc",
          "limit": C.ACC_LIMIT,
          "bias_range": C.ACC_BIAS,
          "scale_range": C.IMU_SCALE_ERROR,
        },
        noise=Unoise(n_min=-n["acc"], n_max=n["acc"]),
      )
    ),
    "joint_pos": _history(
      ObservationTermCfg(
        func=envs_mdp.joint_pos_rel,
        params={"biased": True},  # Encoder bias (DR) is visible to the actor.
        noise=Unoise(n_min=-n["joint_pos"], n_max=n["joint_pos"]),
      )
    ),
    "joint_vel": _history(
      ObservationTermCfg(
        func=envs_mdp.joint_vel_rel,
        noise=Unoise(n_min=-n["joint_vel"], n_max=n["joint_vel"]),
      )
    ),
    "actions": _history(ObservationTermCfg(func=envs_mdp.last_action), delay=False),
    "joint_torque": _history(
      ObservationTermCfg(
        func=mdp.joint_torque,
        noise=Unoise(n_min=-n["torque"], n_max=n["torque"]),
      )
    ),
  }
  # Non-history actor terms: 7 values.
  actor_now = {
    "attitude": ObservationTermCfg(
      func=mdp.attitude_estimate,
      params={"corrupt": True},
      noise=Unoise(n_min=-n["gravity"], n_max=n["gravity"]),
    ),
    "flip_command": ObservationTermCfg(func=mdp.flip_command_obs),
  }

  critic = {
    "gyro": ObservationTermCfg(func=envs_mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "acc": ObservationTermCfg(func=envs_mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_acc"}),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "joint_torque": ObservationTermCfg(func=mdp.joint_torque),
    "attitude": ObservationTermCfg(func=mdp.attitude_estimate, params={"corrupt": False}),
    "flip_command": ObservationTermCfg(func=mdp.flip_command_obs),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel_w),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel_b),
    "flip_state": ObservationTermCfg(func=mdp.flip_state),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_force": ObservationTermCfg(
      func=mdp.foot_contact_force, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={
        "foot_radius": FOOT_RADIUS,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_NAMES),
      },
    ),
    "dr_params": ObservationTermCfg(
      func=mdp.dr_params,
      params={
        "foot_geom_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
        "base_body_cfg": SceneEntityCfg("robot", body_names=(GO2_SPEC.base_body,)),
      },
    ),
  }
  return {
    "actor": ObservationGroupCfg(
      terms={**actor_history, **actor_now}, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic, concatenate_terms=True, enable_corruption=False
    ),
  }


def _events(dr_ranges: dict) -> dict[str, EventTermCfg]:
  r = dr_ranges
  return {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-3.14, 3.14)}, "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.03, 0.03),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="reset",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": r["encoder_bias"]},
    ),
    "foot_friction": EventTermCfg(
      mode="reset",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
        "operation": "abs",
        "axes": [0],
        "ranges": r["foot_friction"],
        "shared_random": True,
      },
    ),
    "added_mass": EventTermCfg(
      mode="reset",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(GO2_SPEC.base_body,)),
        "operation": "add",
        "ranges": r["added_mass"],
      },
    ),
    "base_com": EventTermCfg(
      mode="reset",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(GO2_SPEC.base_body,)),
        "operation": "add",
        "ranges": {i: r["com_offset"] for i in range(3)},
      },
    ),
    "pd_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": r["kp_scale"],
        "kd_range": r["kd_scale"],
        "operation": "scale",
      },
    ),
    # Must stay the last reset event (after encoder_bias); see the docstring.
    "hold_default_joint_targets": EventTermCfg(
      func=mdp.hold_default_joint_targets, mode="reset"
    ),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  """Genesis-backflip style reward set, gated by the time since the trigger
  (mdp/rewards.py). Weights in constants.REWARD_WEIGHTS."""
  w = C.REWARD_WEIGHTS
  R = mdp.rewards
  feet = SceneEntityCfg("robot", site_names=FOOT_NAMES)

  def term(name, func, **params):
    return RewardTermCfg(func=func, weight=w[name], params=params)

  return {
    # Flip (time-gated).
    "flip_orientation": term("flip_orientation", R.flip_orientation),
    "flip_ang_vel": term("flip_ang_vel", R.flip_ang_vel),
    "flip_lin_vel_z": term("flip_lin_vel_z", R.flip_lin_vel_z),
    "feet_height_before": term(
      "feet_height_before", R.feet_height_before, foot_radius=FOOT_RADIUS, asset_cfg=feet
    ),
    "height_control": term("height_control", R.height_control),
    # Always on.
    "gravity_y": term("gravity_y", R.gravity_y),
    "ang_vel_z": term("ang_vel_z", R.ang_vel_z),
    "actions_symmetry": term("actions_symmetry", R.actions_symmetry),
    "feet_distance": term("feet_distance", R.feet_distance, asset_cfg=feet),
    "action_rate": term("action_rate", envs_mdp.action_rate_l2),
    "joint_limits": term("joint_limits", envs_mdp.joint_pos_limits),
  }


def failure_terminations() -> dict[str, TerminationTermCfg]:
  """TM-2 .. TM-7 (terminal failures at the current level). Not used in
  training since 2026-11-15; add them to evaluate a policy against the spec's
  failure conditions."""
  T = TerminationTermCfg
  return {
    "base_contact": T(func=mdp.base_contact, params={"sensor_name": "base_ground_contact"}),  # TM-2
    "base_too_low": T(func=mdp.base_too_low),  # TM-3
    "forward_rotation": T(func=mdp.forward_rotation),  # TM-4
    "over_rotation": T(func=mdp.over_rotation),  # TM-5
    "under_rotated_landing": T(func=mdp.under_rotated_landing),  # TM-6
    "phase_timeout": T(func=mdp.phase_timeout),  # TM-7
  }


def _terminations() -> dict[str, TerminationTermCfg]:
  """No failure terminations in training (as Genesis-backflip): they made
  not trying pay (runs 1-4). Only a broken simulation ends an episode."""
  T = TerminationTermCfg
  return {
    # unstable_sim runs first: it advances the flip state machine.
    "unstable_sim": T(func=mdp.unstable_sim),  # TM-8
    # Truncations (time_out=True): the value bootstraps past them.
    "time_out": T(func=envs_mdp.time_out, time_out=True),  # TM-1
    "success_tail": T(func=mdp.success_tail_elapsed, time_out=True),  # TM-9
    "flip_tail": T(func=mdp.flip_tail_elapsed, time_out=True),  # TM-10
  }


def make_backflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      sensors=_sensors(),
      num_envs=1,
      extent=2.0,
    ),
    observations=_observations(),
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=C.ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands={"flip": mdp.FlipCommandCfg()},
    events=_events(C.DR_NARROW),
    rewards=_rewards(),
    terminations=_terminations(),
    curriculum={
      # Stage / level / reward / DR schedule (mdp/curriculum.py).
      "flip": CurriculumTermCfg(func=mdp.flip_curriculum, params=dict(C.CURRICULUM)),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=GO2_SPEC.base_body,
      distance=2.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50),
    ),
    decimation=4,
    episode_length_s=C.EPISODE_LENGTH_S,
  )

  if play:
    cfg.curriculum = {}
    cfg.observations["actor"].enable_corruption = False
    flip = cfg.commands["flip"]
    assert isinstance(flip, mdp.FlipCommandCfg)
    flip.no_trigger_prob = 0.0
    flip.fixed_stage = len(C.STAGES) - 1
    for name in ("encoder_bias", "foot_friction", "added_mass", "base_com", "pd_gains"):
      cfg.events.pop(name, None)
    # Keep hold_default_joint_targets last.
    cfg.events["hold_default_joint_targets"] = cfg.events.pop("hold_default_joint_targets")
  return cfg
