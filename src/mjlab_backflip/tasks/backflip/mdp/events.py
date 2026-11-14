"""Event terms for the backflip task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def hold_default_joint_targets(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Set the joint position targets of reset envs to the default pose.

  Workaround for the actuator delay (mjlab 1.6.0): after a reset the env calls
  ``scene.write_data_to_sim()`` before any action, so the first frame pushed
  into the freshly reset delay buffer — and backfilled over its whole history
  — is the stale joint target (zeros on the first reset, the previous
  episode's last target on auto-resets). With a 4-substep delay the robot
  then tracks that target for 0.02 s and hops ~8 cm. Run this after the
  encoder bias event so the target matches what a zero action produces.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  target = asset.data.default_joint_pos[env_ids] - asset.data.encoder_bias[env_ids]
  asset.set_joint_position_target(target, env_ids=env_ids)
