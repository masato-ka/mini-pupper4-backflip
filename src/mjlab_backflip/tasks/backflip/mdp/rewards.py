"""Reward terms for the backflip task.

Structure (spec): reward = common terms + phase terms, where a phase term is
``mask(phase in phases) * term``. ``phase_gated`` does the masking; the term
functions below compute the raw value for every env.

Conventions:
  - Every function returns a non-negative magnitude; the weight carries the
    sign (tests/test_rewards.py checks the signs against the spec table).
  - The reward manager multiplies by ``step_dt``. Sparse terms (fired once
    per episode) return ``1 / step_dt`` on that step so their episode total
    equals the weight.
  - Terms marked [cmd] read their target from the latched jump command.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

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
  term.update_state()
  return term


def kern(x: torch.Tensor, std: float) -> torch.Tensor:
  return torch.exp(-torch.square(x / std))


def phase_gated(
  env: ManagerBasedRlEnv,
  phases: tuple[int, ...],
  fn: Callable[..., torch.Tensor],
  command_name: str = "flip",
  **kwargs,
) -> torch.Tensor:
  """fn(env, **kwargs) where the flip phase is in ``phases``, else 0."""
  flip = _flip(env, command_name)
  mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  for p in phases:
    mask |= flip.phase == p
  return torch.where(mask, fn(env, **kwargs), 0.0)


##
# Common.
##


class actions_symmetry:
  """Left-right mirror error of the actions: hips mirrored, thighs/calves equal."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene["robot"]
    names = list(asset.actuator_names)
    left, right, sign = [], [], []
    for i, name in enumerate(names):
      if name[1] != "L":
        continue
      j = names.index(name[0] + "R" + name[2:])
      left.append(i)
      right.append(j)
      sign.append(-1.0 if "hip" in name else 1.0)
    self._left = torch.tensor(left, device=env.device)
    self._right = torch.tensor(right, device=env.device)
    self._sign = torch.tensor(sign, device=env.device)

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    a = env.action_manager.action
    diff = a[:, self._left] - self._sign * a[:, self._right]
    return torch.sum(torch.square(diff), dim=-1)


def terminated_once(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sparse: 1 / step_dt on the failure step, so the penalty is the weight
  once (mjlab's is_terminated would be scaled down to weight * dt).
  Truncations (time_out, success tail) are not failures."""
  return env.termination_manager.terminated.float() / env.step_dt


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 per step while not failed (weight = reward per second of survival).
  Applied in Stand and Done only (phase_gated in the env cfg)."""
  return (~env.termination_manager.terminated).float()


def roll(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """g_y^2."""
  return torch.square(env.scene[asset_cfg.name].data.projected_gravity_b[:, 1])


def yaw_rate(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """omega_z^2 (body frame)."""
  return torch.square(env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, 2])


def contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Number of matched geoms in contact (thigh / calf against the ground)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float().sum(dim=-1)


##
# Stand.
##


def stand_height(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """(h - h_nom)^2."""
  flip = _flip(env, command_name)
  return torch.square(flip.base_height() - flip.spec.h_nom)


def pose_deviation(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """||q - q_default||^2."""
  data = env.scene[asset_cfg.name].data
  return torch.sum(torch.square(data.joint_pos - data.default_joint_pos), dim=-1)


def body_motion(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """||v||^2 + 0.1 ||omega||^2."""
  data = env.scene[asset_cfg.name].data
  v = torch.sum(torch.square(data.root_link_lin_vel_w), dim=-1)
  w = torch.sum(torch.square(data.root_link_ang_vel_b), dim=-1)
  return v + 0.1 * w


class feet_distance:
  """Squared deviation of the feet's lateral offsets (body frame) from the
  default stance. Gated to ``phases`` itself (class terms cannot be wrapped by
  ``phase_gated``)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._site_ids = cfg.params["asset_cfg"].site_ids
    # Nominal |y| per foot, taken from the first call (the reset stance).
    self._y_nom: torch.Tensor | None = None

  def _feet_y(self, asset) -> torch.Tensor:
    d = asset.data
    rel = d.site_pos_w[:, self._site_ids] - d.root_link_pos_w[:, None]
    q = d.root_link_quat_w[:, None].expand(-1, rel.shape[1], -1)
    return quat_apply_inverse(q, rel)[..., 1]

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    phases: tuple[int, ...] = (C.STAND,),
    command_name: str = "flip",
  ) -> torch.Tensor:
    flip = _flip(env, command_name)
    y = self._feet_y(env.scene[asset_cfg.name]).abs()
    if self._y_nom is None:
      self._y_nom = y.mean(dim=0)
    err = torch.sum(torch.square(y - self._y_nom), dim=-1)
    mask = torch.zeros_like(err, dtype=torch.bool)
    for p in phases:
      mask |= flip.phase == p
    return torch.where(mask, err, 0.0)


##
# Crouch.
##


def crouch_depth(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """[cmd] Sparse: kern(h - rho_c h_nom, 0.03) once, on the Crouch -> Launch
  step (T2).

  Was dense in Crouch; the kernel is centred on the T2 threshold, so the
  policy hovered 1.5 cm above it and farmed the term for the whole Crouch
  time limit (4th Colab run, 2026-11-14).
  """
  flip = _flip(env, command_name)
  value = kern(flip.base_height() - flip.crouch_height, C.CROUCH_DEPTH_STD)
  return torch.where(flip.entered[:, C.LAUNCH], value / env.step_dt, 0.0)


def pitch_tilt(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """g_x^2 (keep the body level)."""
  return torch.square(env.scene[asset_cfg.name].data.projected_gravity_b[:, 0])


def feet_lift(
  env: ManagerBasedRlEnv,
  foot_radius: float,
  threshold: float = C.FEET_STAY_HEIGHT,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """sum clamp(z_foot - 0.02, min=0) (keep the feet on the ground)."""
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - env.scene.env_origins[:, 2:3]
  return torch.sum((z - foot_radius - threshold).clamp(min=0.0), dim=-1)


def upward_velocity_sq(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """relu(v_z)^2 (no jumping while crouching)."""
  vz = env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2]
  return torch.square(vz.clamp(min=0.0))


##
# Launch.
##


def takeoff_vz(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """[cmd] clip(v_z, 0, v*) / v*."""
  flip = _flip(env, command_name)
  vz = flip.robot.data.root_link_lin_vel_w[:, 2]
  v_star = flip.v_star
  return torch.minimum(vz.clamp(min=0.0), v_star) / v_star


def vz_overshoot(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """[cmd] relu(v_z - 1.1 v*)^2."""
  flip = _flip(env, command_name)
  vz = flip.robot.data.root_link_lin_vel_w[:, 2]
  return torch.square((vz - C.VZ_OVERSHOOT_RATIO * flip.v_star).clamp(min=0.0))


def pitch_kick(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """[cmd] clip(omega_back, 0, omega*) / omega*."""
  flip = _flip(env, command_name)
  w_star = flip.omega_star
  return torch.minimum(flip.omega_back().clamp(min=0.0), w_star) / w_star


def rear_before_front(
  env: ManagerBasedRlEnv,
  front_ids: tuple[int, ...],
  rear_ids: tuple[int, ...],
  command_name: str = "flip",
) -> torch.Tensor:
  """Fraction of rear feet off the ground while a front foot still pushes.

  In a backflip the front feet leave first and the body pivots on the rear
  feet (Stage-Wise CMORL).
  """
  flip = _flip(env, command_name)
  contact = flip.foot_contact()
  rear_off = (~contact[:, list(rear_ids)]).float().mean(dim=-1)
  front_on = contact[:, list(front_ids)].any(dim=-1)
  return rear_off * front_on.float()


##
# Air.
##


def apex_bonus(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """[cmd] Sparse: kern(H - dh*, 0.06) once, at the apex."""
  flip = _flip(env, command_name)
  value = kern(flip.H_max - flip.target_height, C.APEX_STD)
  return torch.where(flip.apex_now, value / env.step_dt, 0.0)


def apex_overshoot(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """[cmd] relu(H - 1.2 dh*)^2 with H = h - h_to."""
  flip = _flip(env, command_name)
  rise = flip.base_height() - flip.h_to
  return torch.square((rise - C.APEX_OVERSHOOT_RATIO * flip.target_height).clamp(min=0.0))


def flip_progress(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """[cmd] kern(theta - theta_ref(tau), 0.8), theta_ref = 2 pi clip(tau / t_f*, 0, 1)."""
  flip = _flip(env, command_name)
  tau = flip.phase_time
  theta_ref = 2 * math.pi * (tau / flip.tf_star).clamp(0.0, 1.0)
  return kern(flip.theta - theta_ref, C.FLIP_PROGRESS_STD)


def wrong_direction(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """Rate of rotation against the backflip direction, relu(-omega_back)."""
  flip = _flip(env, command_name)
  return (-flip.omega_back()).clamp(min=0.0)


class tuck:
  """||q - q_tuck||^2 in Air while theta is in [0.25 pi, 1.5 pi] (optional term).
  Gated to Air itself (class terms cannot be wrapped by ``phase_gated``)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene["robot"]
    q = asset.data.default_joint_pos[0].clone()
    for i, name in enumerate(asset.joint_names):
      for key, offset in C.TUCK_OFFSETS.items():
        if key in name:
          q[i] += offset
    lo, hi = asset.data.soft_joint_pos_limits[0].unbind(-1)
    self._q_tuck = torch.minimum(torch.maximum(q, lo), hi)

  def reset(self, env_ids=None) -> None:
    pass

  def __call__(self, env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
    flip = _flip(env, command_name)
    lo, hi = C.TUCK_THETA_RANGE
    active = (flip.phase == C.AIR) & (flip.theta >= lo) & (flip.theta <= hi)
    err = torch.sum(torch.square(flip.robot.data.joint_pos - self._q_tuck), dim=-1)
    return torch.where(active, err, 0.0)


def land_prep(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """kern(||q - q_default||, 0.5) once theta > 1.5 pi (optional term)."""
  flip = _flip(env, command_name)
  data = flip.robot.data
  dist = torch.linalg.norm(data.joint_pos - data.default_joint_pos, dim=-1)
  return torch.where(flip.theta > C.LAND_PREP_THETA, kern(dist, C.LAND_PREP_STD), 0.0)


##
# Land (positive terms multiplied by f_flip).
##


def upright(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """f_flip * kern(||g_b - g_down||, 0.3)."""
  flip = _flip(env, command_name)
  g = flip.robot.data.projected_gravity_b
  g_down = torch.tensor([0.0, 0.0, -1.0], device=env.device)
  return flip.f_flip * kern(torch.linalg.norm(g - g_down, dim=-1), C.UPRIGHT_STD)


def land_height(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """f_flip * kern(h - h_nom, 0.05)."""
  flip = _flip(env, command_name)
  return flip.f_flip * kern(flip.base_height() - flip.spec.h_nom, C.LAND_HEIGHT_STD)


def feet_all_contact(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """f_flip * n_c / 4."""
  flip = _flip(env, command_name)
  contact = flip.foot_contact()
  return flip.f_flip * contact.float().mean(dim=-1)


def success_bonus(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """Sparse: once, on the step the success is latched."""
  flip = _flip(env, command_name)
  return flip.success_now.float() / env.step_dt
