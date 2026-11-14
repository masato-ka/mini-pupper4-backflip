"""Termination terms (TM-2 .. TM-9). Thresholds follow the current level.

Every term calls ``flip.update_state()`` first: terminations are the first
manager to run after physics, so this is where the state machine advances
for the step (see flip_command.py).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_backflip.tasks.backflip import constants as C
from mjlab_backflip.tasks.backflip.mdp.flip_command import FlipCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor

_ROBOT = SceneEntityCfg("robot")


def _flip(env: ManagerBasedRlEnv, command_name: str) -> FlipCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, FlipCommand)
  term.update_state()
  return term


def base_contact(
  env: ManagerBasedRlEnv, sensor_name: str, command_name: str = "flip"
) -> torch.Tensor:
  """TM-2: torso touches the ground (any phase)."""
  _flip(env, command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).any(dim=-1)


def base_too_low(
  env: ManagerBasedRlEnv, grace_steps: int = 20, command_name: str = "flip"
) -> torch.Tensor:
  """TM-3: base below the level's minimum height, after a grace period."""
  flip = _flip(env, command_name)
  low = flip.base_height() < flip.level_cfg.min_height
  return low & (env.episode_length_buf > grace_steps)


def forward_rotation(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """TM-4: rotated the wrong way (front flip)."""
  flip = _flip(env, command_name)
  return flip.theta < -flip.level_cfg.forward_rotation_limit


def over_rotation(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """TM-5: rotated past one full turn by more than the margin."""
  flip = _flip(env, command_name)
  return flip.theta > 2 * math.pi + flip.level_cfg.over_rotation_margin


def under_rotated_landing(
  env: ManagerBasedRlEnv, command_name: str = "flip"
) -> torch.Tensor:
  """TM-6: touched down with f_flip below the level's minimum."""
  flip = _flip(env, command_name)
  limit = flip.level_cfg.min_f_flip_at_touchdown
  if limit is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  return flip.entered[:, C.LAND] & (flip.f_flip < limit)


def phase_timeout(env: ManagerBasedRlEnv, command_name: str = "flip") -> torch.Tensor:
  """TM-7: stayed too long in Crouch / Launch / Air / Land (Stand has no limit)."""
  flip = _flip(env, command_name)
  scale = flip.level_cfg.phase_time_scale
  out = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  for phase, limit in C.PHASE_TIME_LIMITS_S.items():
    out |= (flip.phase == phase) & (flip.phase_time > limit * scale)
  return out


def unstable_sim(
  env: ManagerBasedRlEnv,
  max_lin_vel: float = 20.0,
  command_name: str = "flip",
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """TM-8: non-finite state or base speed above max_lin_vel."""
  _flip(env, command_name)
  data = env.scene[asset_cfg.name].data
  state = torch.cat(
    (data.root_link_pos_w, data.root_link_lin_vel_w, data.root_link_ang_vel_w, data.joint_vel),
    dim=-1,
  )
  bad = ~torch.isfinite(state).all(dim=-1)
  return bad | (data.root_link_lin_vel_w.norm(dim=-1) > max_lin_vel)


def success_tail_elapsed(
  env: ManagerBasedRlEnv, command_name: str = "flip"
) -> torch.Tensor:
  """TM-9 (time out): SUCCESS_TAIL_S has passed since success."""
  flip = _flip(env, command_name)
  t = env.episode_length_buf.float() * env.step_dt
  return flip.success & (t - flip.success_time >= C.SUCCESS_TAIL_S)
