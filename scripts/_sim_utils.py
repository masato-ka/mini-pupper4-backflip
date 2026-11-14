"""Helpers shared by the check scripts: a deterministic, headless Go2 env."""

import mjlab  # noqa: F401  (loads the mjlab.tasks entry points)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

BASE_TASK = "Mjlab-Velocity-Flat-Unitree-Go2"


def make_plain_env(
  num_envs: int = 1, device: str = "cpu", commands: dict | None = None
) -> ManagerBasedRlEnv:
  """Go2 on flat ground with no randomization, pushes or terminations.

  ``commands`` are added next to the velocity task's ``twist`` command.
  """
  cfg = load_env_cfg(BASE_TASK, play=True)
  cfg.commands.update(commands or {})
  cfg.scene.num_envs = num_envs
  for name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(name, None)
  # Reset exactly to the init state (no pose offsets).
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.terminations = {}
  cfg.observations["actor"].enable_corruption = False
  return ManagerBasedRlEnv(cfg=cfg, device=device)
