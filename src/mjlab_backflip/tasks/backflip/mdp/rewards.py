"""Reward terms for the backflip task (revised 2026-11-15).

The flip terms are gated by tau, the time since the accepted trigger, on the
Genesis-backflip timeline (constants.py): feet down before the flip, upward
velocity in the lift window, backward pitch rate in the rotation window, a
reference attitude that turns one full backflip over the rotation window,
the standing height before the trigger and after the flip, and (added to
the Genesis set) a landing term for standing after exactly one turn. Without a
trigger (control episodes) tau = -inf, so only the standing terms apply.

The phase state machine is not used for gating any more: every phase-gated
version (four Colab runs, 2026-11-14) was beaten by a loophole in the
gating. tau is in the actor's observation (flip_command), so the actor can
tell the windows apart.

Conventions:
  - The weight carries the sign. Raw values are non-negative except
    flip_ang_vel and flip_lin_vel_z, which are signed as in Genesis
    (rotating or moving the wrong way in the window costs).
  - The reward manager multiplies by ``step_dt``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_backflip.tasks.backflip import constants as C
from mjlab_backflip.tasks.backflip.mdp.flip_command import FlipCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _flip(env: ManagerBasedRlEnv, command_name: str = "flip") -> FlipCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, FlipCommand)
  term.update_state()
  return term


def kern(x: torch.Tensor, std: float) -> torch.Tensor:
  return torch.exp(-torch.square(x / std))


def _in_window(tau: torch.Tensor, start: float, length: float) -> torch.Tensor:
  return (tau >= start) & (tau < start + length)


##
# Flip (time-gated).
##


def reference_gravity(flip: FlipCommand) -> torch.Tensor:
  """g_ref(tau): gravity in the body frame after pitching back by
  theta_ref = 2 pi clip((tau - start) / rotate, 0, 1). Upright before and
  after the rotation window."""
  tau = flip.time_since_trigger()
  frac = ((tau - C.FLIP_START_S) / C.FLIP_ROTATE_S).clamp(0.0, 1.0)
  theta_ref = 2 * math.pi * frac
  # Inverse of RobotSpec.back_pitch_angle: atan2(g_x, -g_z) = sign * theta.
  g_x = flip.spec.back_pitch_sign * torch.sin(theta_ref)
  return torch.stack((g_x, torch.zeros_like(g_x), -torch.cos(theta_ref)), dim=-1)


def flip_orientation(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """||g_b - g_ref(tau)||^2 (Genesis orientation_control)."""
  flip = _flip(env, command_name)
  g = flip.robot.data.projected_gravity_b
  return torch.sum(torch.square(g - reference_gravity(flip)), dim=-1)


def flip_ang_vel(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """clip(omega_back, -7.2, 7.2) in the rotation window (Genesis ang_vel_y)."""
  flip = _flip(env, command_name)
  w = flip.omega_back().clamp(-C.ANG_VEL_CLIP, C.ANG_VEL_CLIP)
  active = _in_window(flip.time_since_trigger(), C.FLIP_START_S, C.FLIP_ROTATE_S)
  return torch.where(active, w, 0.0)


def flip_lin_vel_z(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """min(v_z, 3) in the lift window (Genesis lin_vel_z)."""
  flip = _flip(env, command_name)
  vz = flip.robot.data.root_link_lin_vel_w[:, 2].clamp(max=C.LIN_VEL_Z_CLIP)
  active = _in_window(flip.time_since_trigger(), C.FLIP_START_S, C.FLIP_LIFT_S)
  return torch.where(active, vz, 0.0)


def feet_height_before(
  env: ManagerBasedRlEnv,
  foot_radius: float,
  command_name: str = "flip",
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """sum relu(z_foot - 0.02) before the flip starts (Genesis
  feet_height_before_backflip): no hopping while standing or crouching."""
  flip = _flip(env, command_name)
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - env.scene.env_origins[:, 2:3]
  lift = torch.sum((z - foot_radius - C.FEET_LIFT_TOLERANCE).clamp(min=0.0), dim=-1)
  return torch.where(flip.time_since_trigger() < C.FLIP_START_S, lift, 0.0)


def height_control(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """(h - h_nom)^2 before the trigger and after the flip (Genesis height_control)."""
  flip = _flip(env, command_name)
  tau = flip.time_since_trigger()
  active = (tau < 0.0) | (tau > C.FLIP_START_S + C.FLIP_SETTLE_S)
  return torch.where(active, torch.square(flip.base_height() - flip.spec.h_nom), 0.0)


def flip_landing(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """kern(theta - 2 pi, 0.5) * kern(h - h_nom, 0.1) * n_c / 4 once the
  rotation window is over: standing on the feet after exactly one turn.

  ~0 without a flip (theta ~ 0) and after over-rotating, so it cannot be
  earned by hopping in place (the Classmethod failure with a large landing
  reward). A continuous function of the state, not a phase flag.
  """
  flip = _flip(env, command_name)
  rot = kern(flip.theta - 2 * math.pi, C.LANDING_THETA_STD)
  height = kern(flip.base_height() - flip.spec.h_nom, C.LANDING_HEIGHT_STD)
  feet = flip.foot_contact().float().mean(dim=-1)
  after = flip.time_since_trigger() >= C.FLIP_START_S + C.FLIP_ROTATE_S
  return torch.where(after, rot * height * feet, 0.0)


def stand_still(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """EV-6 filtered |omega_base| + mean |q_dot| before the accepted trigger
  (the whole episode without one).

  The interlock that waits for these to settle is off in training (the
  exploration noise never lets them settle); this term teaches the still
  stand it needs on the real robot. Before the trigger, so it never weighs
  on the flip itself.
  """
  flip = _flip(env, command_name)
  return torch.where(flip.triggered, 0.0, flip.filt_ang_vel + flip.filt_joint_vel)


##
# Always on.
##


def gravity_y(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """g_y^2 (roll away from the sagittal plane)."""
  return torch.square(env.scene[asset_cfg.name].data.projected_gravity_b[:, 1])


def ang_vel_z(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """|omega_z| (body frame)."""
  return env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, 2].abs()


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


class feet_distance:
  """Squared deviation of the feet's lateral offsets (body frame) from the
  default stance. Genesis' version pulls the feet to y = 0 (its stance width
  is multiplied by zeros); this one keeps the nominal stance."""

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
  ) -> torch.Tensor:
    y = self._feet_y(env.scene[asset_cfg.name]).abs()
    if self._y_nom is None:
      self._y_nom = y.mean(dim=0)
    return torch.sum(torch.square(y - self._y_nom), dim=-1)
