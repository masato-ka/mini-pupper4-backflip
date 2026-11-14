"""Observation terms for the backflip task.

Actor terms only use what the real robot can measure (IMU, encoders, motor
torque, the operator's command and trigger). Sensor errors that persist over
an episode (IMU bias / scale, estimator drift) are modelled inside class terms
and resampled on reset; per-step white noise uses the observation manager's
``noise`` so it follows the group's ``enable_corruption``.

Ordering note: ``actuator_force`` and the action vector are in actuator order
(all hips, all thighs, all calves); joint observations are in joint order
(per leg). See docs/notes.md.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz

from mjlab_backflip.tasks.backflip import constants as C
from mjlab_backflip.tasks.backflip.mdp.flip_command import FlipCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor

_ROBOT = SceneEntityCfg("robot")


def _flip(env: ManagerBasedRlEnv, command_name: str = "flip") -> FlipCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, FlipCommand)
  return term


##
# Actor.
##


class imu_reading(ManagerTermBase):
  """IMU channel with per-episode bias and scale error, then saturation.

  y = clip((1 + s) * x + b, -limit, limit), s ~ U(scale_range), b ~ U(bias_range)
  per env and axis, resampled on reset.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    p = cfg.params
    self._sensor_name: str = p["sensor_name"]
    self._bias_range = p.get("bias_range", (0.0, 0.0))
    self._scale_range = p.get("scale_range", (0.0, 0.0))
    self._bias = torch.zeros(env.num_envs, 3, device=env.device)
    self._scale = torch.zeros(env.num_envs, 3, device=env.device)
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._bias[ids] = torch.empty_like(self._bias[ids]).uniform_(*self._bias_range)
    self._scale[ids] = torch.empty_like(self._scale[ids]).uniform_(*self._scale_range)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    limit: float,
    bias_range: tuple[float, float] = (0.0, 0.0),
    scale_range: tuple[float, float] = (0.0, 0.0),
  ) -> torch.Tensor:
    x = env.scene[sensor_name].data
    return ((1.0 + self._scale) * x + self._bias).clamp(-limit, limit)


def joint_torque(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Motor torque (or current-based estimate), actuator order."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.actuator_force[:, asset_cfg.actuator_ids]


class attitude_estimate(ManagerTermBase):
  """Stand-in for the on-board estimator: [g_est (3), theta_est / 2 pi (1)].

  True gravity direction rotated by a slowly drifting tilt bias, and true
  theta plus a drifting bias. Bias random walks with std ``drift_std`` per
  sqrt(s), reset to a small random value each episode. Replace with the real
  estimator (gyro propagation, accelerometer correction when |a| ~ g) before
  transfer.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    p = cfg.params
    self._init_bias = p.get("init_bias", 0.02)
    self._drift_std = p.get("drift_std", 0.01)
    self._tilt = torch.zeros(env.num_envs, 2, device=env.device)  # roll, pitch [rad]
    self._theta_bias = torch.zeros(env.num_envs, device=env.device)
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    b = self._init_bias
    self._tilt[ids] = torch.empty_like(self._tilt[ids]).uniform_(-b, b)
    self._theta_bias[ids] = torch.empty_like(self._theta_bias[ids]).uniform_(-b, b)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "flip",
    corrupt: bool = True,
    init_bias: float = 0.02,
    drift_std: float = 0.01,
    asset_cfg: SceneEntityCfg = _ROBOT,
  ) -> torch.Tensor:
    flip = _flip(env, command_name)
    flip.update_state()
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    theta = flip.theta
    if corrupt:
      step_std = self._drift_std * math.sqrt(env.step_dt)
      self._tilt += torch.randn_like(self._tilt) * step_std
      self._theta_bias += torch.randn_like(self._theta_bias) * step_std
      zero = torch.zeros_like(self._theta_bias)
      q = quat_from_euler_xyz(self._tilt[:, 0], self._tilt[:, 1], zero)
      g = quat_apply_inverse(q, g)
      theta = theta + self._theta_bias
    return torch.cat((g, (theta / (2 * math.pi)).unsqueeze(-1)), dim=-1)


def flip_command_obs(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """[command scalar, trigger flag, time since trigger / 2 s] (3)."""
  flip = _flip(env, command_name)
  flip.update_state()
  return flip.command


##
# Critic (privileged).
##


def base_lin_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_lin_vel_w


def base_ang_vel_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_ang_vel_b


def flip_state(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """True flip state (17):
  h, theta / 2 pi, phase one-hot (6), phase time, air time, h_to, H,
  v*, t_f*, f_flip, triggered, dh*.
  """
  flip = _flip(env, command_name)
  flip.update_state()
  h = flip.base_height()
  in_air = flip.phase == C.AIR
  rise = torch.where(in_air, h - flip.h_to, 0.0)
  air_time = torch.where(in_air, flip.phase_time, 0.0)
  onehot = torch.nn.functional.one_hot(flip.phase, C.NUM_PHASES).float()
  cols = (
    h,
    flip.theta / (2 * math.pi),
    flip.phase_time,
    air_time,
    flip.h_to,
    rise,
    flip.v_star,
    flip.tf_star,
    flip.f_flip,
    flip.triggered.float(),
    flip.target_height,
  )
  return torch.cat((torch.stack(cols[:2], -1), onehot, torch.stack(cols[2:], -1)), -1)


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()


def foot_contact_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Normal (world z) contact force per foot, scaled to body weights."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  return sensor.data.force[..., 2] / 150.0


def foot_height(
  env: ManagerBasedRlEnv, foot_radius: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Foot sphere bottom height above the ground (flat terrain)."""
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  return z - env.scene.env_origins[:, 2:3] - foot_radius


def _per_env(field: torch.Tensor, num_envs: int) -> torch.Tensor:
  # Model fields are per-world only once a DR term expanded them.
  return field if field.shape[0] == num_envs and field.dim() >= 2 else field.expand(
    num_envs, *field.shape
  )


def dr_params(
  env: ManagerBasedRlEnv,
  foot_geom_cfg: SceneEntityCfg,
  base_body_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Domain randomization parameters (19):
  foot friction (1), added base mass (1), base COM offset (3),
  mean Kp scale (1), mean Kd scale (1), encoder bias (12).
  """
  n = env.num_envs
  model = env.sim.model
  asset: Entity = env.scene[asset_cfg.name]
  # SceneEntityCfg ids are entity-local; model fields need global ids.
  geom_ids = asset.indexing.geom_ids[foot_geom_cfg.geom_ids]
  body_id = asset.indexing.body_ids[base_body_cfg.body_ids]
  friction = _per_env(model.geom_friction, n)[:, geom_ids, 0].mean(-1, keepdim=True)
  mass = _per_env(model.body_mass, n)[:, body_id].reshape(n, -1)[:, :1]
  mass0 = env.sim.get_default_field("body_mass")[body_id].reshape(-1)[:1]
  ipos = _per_env(model.body_ipos, n)[:, body_id].reshape(n, -1)[:, :3]
  ipos0 = env.sim.get_default_field("body_ipos")[body_id].reshape(-1)[:3]
  ctrl_ids = asset.indexing.ctrl_ids
  gain = _per_env(model.actuator_gainprm, n)[:, ctrl_ids, 0]
  bias_kd = _per_env(model.actuator_biasprm, n)[:, ctrl_ids, 2]
  gain0 = env.sim.get_default_field("actuator_gainprm")[ctrl_ids, 0]
  kd0 = env.sim.get_default_field("actuator_biasprm")[ctrl_ids, 2]
  kp_scale = (gain / gain0).mean(-1, keepdim=True)
  kd_scale = (bias_kd / kd0).mean(-1, keepdim=True)
  return torch.cat(
    (friction, mass - mass0, ipos - ipos0, kp_scale, kd_scale, asset.data.encoder_bias), -1
  )
