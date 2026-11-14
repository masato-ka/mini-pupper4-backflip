"""Stage / level curriculum (M5), driven with synthetic episode outcomes."""

import math

import pytest
import torch

import mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from mjlab_backflip.tasks.backflip import constants as C
from mjlab_backflip.tasks.backflip.mdp.curriculum import LAND_REWARDS

TASK = "Mjlab-Backflip-Flat-Unitree-Go2"
FAST = {"promote_iters": 3, "demote_iters": 3, "min_iters": 5, "success_ramp_iters": 10,
        "ema_episodes": 2}


@pytest.fixture()
def env():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 8
  cfg.curriculum["flip"].params.update(FAST)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  return env


def _term(env):
  return env.curriculum_manager._term_cfgs[0].func


def _params(env):
  return env.curriculum_manager._term_cfgs[0].params


def _finish_episodes(env, rotated: bool, success: bool, cmd=None):
  """Fake one batch of finished, triggered episodes and run the curriculum."""
  flip = env.command_manager.get_term("flip")
  ids = torch.arange(env.num_envs)
  env.episode_length_buf[:] = 100
  flip.triggered[:] = True
  flip.max_phase[:] = C.DONE if rotated else C.AIR
  flip.theta_td[:] = 2 * math.pi if rotated else 0.0
  flip.success[:] = success
  if cmd is not None:
    flip.cmd_idx[:] = cmd
  env.common_step_counter += 24  # One policy iteration.
  return _term(env)(env, ids, **_params(env))


def _stage(state):
  return int(state["stage"]) - 1


def _until_stage(env, target, max_iters=20, cmds=(None,), **outcome):
  """Run iterations until the curriculum reaches ``target`` (or give up)."""
  state = None
  for _ in range(max_iters):
    for cmd in cmds:
      state = _finish_episodes(env, cmd=cmd, **outcome)
    if _stage(state) == target:
      break
  return state


def test_initial_stage_effects(env):
  rm = env.reward_manager
  flip = env.command_manager.get_term("flip")
  assert flip.stage == 0 and flip.level == 0
  assert flip.allowed_commands == (C.MID,)
  for name in LAND_REWARDS:
    assert rm.get_term_cfg(name).weight == 0.0  # Land terms off in S1.
  assert rm.get_term_cfg("success_bonus").weight == 0.0
  assert env.event_manager.get_term_cfg("foot_friction").params["ranges"] == C.DR_NARROW["foot_friction"]


def test_promotion_ladder_and_effects(env):
  rm = env.reward_manager
  flip = env.command_manager.get_term("flip")
  # S1 -> S2 on rotation rate.
  state = _until_stage(env, 1, rotated=True, success=False)
  assert _stage(state) == 1
  for name in LAND_REWARDS:
    assert rm.get_term_cfg(name).weight == C.REWARD_WEIGHTS[name]
  # success_bonus ramps from 0 in S2.
  w0 = rm.get_term_cfg("success_bonus").weight
  assert 0.0 <= w0 < C.REWARD_WEIGHTS["success_bonus"]
  # S2 -> S3 on Mid success.
  state = _until_stage(env, 2, cmds=(C.MID,), rotated=True, success=True)
  assert _stage(state) == 2
  assert flip.level == 1 and flip.allowed_commands == (C.LOW, C.MID, C.HIGH)
  assert rm.get_term_cfg("success_bonus").weight == C.REWARD_WEIGHTS["success_bonus"]
  assert env.event_manager.get_term_cfg("foot_friction").params["ranges"] == C.DR_FULL["foot_friction"]
  assert env.event_manager.get_term_cfg("pd_gains").params["kp_range"] == C.DR_FULL["kp_scale"]
  # S3 -> S4 needs every command; Mid alone is not enough.
  state = _until_stage(env, 3, cmds=(C.MID,), rotated=True, success=True)
  assert _stage(state) == 2
  state = _until_stage(env, 3, cmds=(C.LOW, C.MID, C.HIGH), rotated=True, success=True)
  assert _stage(state) == 3 and flip.level == 2


def test_demotion_after_min_stay(env):
  state = _until_stage(env, 1, rotated=True, success=False)
  assert _stage(state) == 1
  # Success stays at 0 in S2: demoted once min_iters and demote_iters pass.
  for _ in range(FAST["min_iters"] - 1):
    state = _finish_episodes(env, rotated=True, success=False)
  assert _stage(state) == 1
  for _ in range(5):
    state = _finish_episodes(env, rotated=True, success=False)
  assert _stage(state) == 0


def test_control_episodes_do_not_count(env):
  flip = env.command_manager.get_term("flip")
  ids = torch.arange(env.num_envs)
  for _ in range(10):
    env.episode_length_buf[:] = 100
    flip.triggered[:] = False  # Control episodes only.
    flip.max_phase[:] = C.DONE
    flip.theta_td[:] = 2 * math.pi
    env.common_step_counter += 24
    state = _term(env)(env, ids, **_params(env))
  assert _stage(state) == 0 and state["rotation_rate"] == 0.0


def test_fixed_stage(env):
  _params(env)["fixed_stage"] = 2
  state = _finish_episodes(env, rotated=False, success=False)
  flip = env.command_manager.get_term("flip")
  assert _stage(state) == 2 and flip.level == 1
