"""Backflip env configuration: dimensions, delays, events, DR visibility (M3)."""

import pytest
import torch

import mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

from mjlab_backflip.tasks.backflip import constants as C

TASK = "Mjlab-Backflip-Flat-Unitree-Go2"


@pytest.fixture(scope="module")
def train_env():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 16
  return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def test_registered():
  assert TASK in list_tasks()


def test_observation_dims(train_env):
  obs, _ = train_env.reset()
  # Actor: 54 per step x 4 history + g_est (3) + theta_est (1) + command (3).
  assert obs["actor"].shape == (16, 54 * C.OBS_HISTORY + 7) == (16, 223)
  assert obs["critic"].shape == (16, 115)
  for _ in range(5):
    obs, *_ = train_env.step(torch.zeros(16, 12))
  assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()


def test_action_delay_and_scale():
  cfg = load_env_cfg(TASK)
  for act in cfg.scene.entities["robot"].articulation.actuators:
    assert act.delay_min_lag == act.delay_max_lag == C.ACTION_DELAY_PHYSICS_STEPS
  assert cfg.actions["joint_pos"].scale == C.ACTION_SCALE
  assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(0.02)  # 50 Hz
  assert cfg.episode_length_s == C.EPISODE_LENGTH_S


def test_default_target_event_runs_last():
  for play in (False, True):
    events = load_env_cfg(TASK, play=play).events
    resets = [k for k, v in events.items() if v.mode == "reset"]
    assert resets[-1] == "hold_default_joint_targets"


def test_no_orientation_termination():
  # The velocity task's bad_orientation would end every flip.
  terms = load_env_cfg(TASK).terminations
  assert "fell_over" not in terms and "illegal_contact" not in terms


def test_dr_params_vary_across_envs(train_env):
  train_env.reset()
  obs = train_env.observation_manager.compute()
  dr = obs["critic"][:, -19:]
  # friction, added mass, COM x/y/z, Kp scale, Kd scale vary per env.
  assert (dr[:, :7].std(dim=0) > 0).all()
  lo, hi = C.DR_NARROW["foot_friction"]
  assert ((dr[:, 0] >= lo - 1e-6) & (dr[:, 0] <= hi + 1e-6)).all()
  lo, hi = C.DR_NARROW["added_mass"]
  assert ((dr[:, 1] >= lo - 1e-6) & (dr[:, 1] <= hi + 1e-6)).all()
