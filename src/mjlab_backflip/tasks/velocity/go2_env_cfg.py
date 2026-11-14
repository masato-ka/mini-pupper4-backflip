"""Unitree Go2 flat velocity environment (mjlab 1.6.0).

Ported from unitreerobotics/unitree_rl_mjlab
(src/tasks/velocity/config/go2/env_cfgs.py, commit 1425b15, Apache-2.0),
rebased on mjlab 1.6.0's ``make_velocity_env_cfg`` and Go1 flat config.
Used as the M0 sanity task and as the base of the backflip task.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg
from mjlab.sensor import TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_backflip.robots import GO2_SPEC, get_go2_robot_cfg

FOOT_NAMES = GO2_SPEC.foot_names
FOOT_GEOM_NAMES = tuple(f"{name}_foot_collision" for name in FOOT_NAMES)


def feet_ground_contact_cfg() -> ContactSensorCfg:
  return ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Flat terrain: no terrain scan, no terrain curriculum.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in FOOT_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_contact_cfg(),)

  # unitree_rl_mjlab keeps the base action scale (0.5).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.5

  cfg.viewer.body_name = GO2_SPEC.base_body
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (GO2_SPEC.base_body,)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = cfg.rewards["pose"].params[
    "std_walking"
  ]
  cfg.rewards["upright"].params["asset_cfg"].body_names = (GO2_SPEC.base_body,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (
    GO2_SPEC.base_body,
  )
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_NAMES

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
