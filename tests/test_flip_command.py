"""State machine and trigger behaviour of FlipCommand (M2), CPU, no learning."""

import math

import pytest
import torch

from _sim_utils import make_plain_env
from check_state_machine import run_scripted_flip, scripted_flip_cfg

from mjlab_backflip.tasks.backflip import constants as C
from mjlab_backflip.tasks.backflip.mdp import FlipCommandCfg


def _step_for(env, seconds, before_step=None):
  act = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  for _ in range(int(round(seconds / env.step_dt))):
    if before_step is not None:
      before_step()
    env.step(act)


@pytest.fixture(scope="module")
def scripted_run():
  env = make_plain_env(commands={"flip": scripted_flip_cfg()})
  phases, _ = run_scripted_flip(env, assist_landing=True)
  return env.command_manager.get_term("flip"), phases


def test_phases_advance_in_order_to_done(scripted_run):
  flip, phases = scripted_run
  diffs = [b - a for a, b in zip(phases, phases[1:])]
  assert all(0 <= d <= 1 for d in diffs)  # Forward only, one step at a time.
  assert [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]] == list(range(6))
  assert bool(flip.success[0])
  assert bool(flip.triggered[0])


def test_latches_after_flip(scripted_run):
  flip, _ = scripted_run
  assert flip.theta_td[0] > math.pi  # Mostly rotated at touchdown.
  assert 0.0 < flip.f_flip[0] <= 1.0
  assert flip.h_to[0] > 0.2
  assert flip.H_max[0] > 0.2


def test_control_episode_never_triggers():
  cfg = scripted_flip_cfg()
  cfg.no_trigger_prob = 1.0
  env = make_plain_env(commands={"flip": cfg})
  flip = env.command_manager.get_term("flip")
  env.reset()
  _step_for(env, 2.0)
  assert not bool(flip.triggered[0])
  assert int(flip.phase[0]) == C.STAND
  assert flip.command[0, 1].item() == 0.0


def test_interlock_waits_for_stable_stand():
  env = make_plain_env(commands={"flip": scripted_flip_cfg()})
  robot = env.scene["robot"]
  flip = env.command_manager.get_term("flip")
  env.reset()
  shake = lambda: robot.write_joint_velocity_to_sim(torch.full_like(robot.data.joint_vel, 3.0))
  # Keep the joints moving until 1.0 s: the 0.5 s trigger must be held back.
  _step_for(env, 1.0, before_step=shake)
  assert not bool(flip.triggered[0])
  _step_for(env, 1.0)
  assert bool(flip.triggered[0])
  assert flip.trigger_time[0] >= 1.0 + C.STABLE_HOLD_S - 1e-6


def test_command_changes_before_trigger_and_latches():
  cfg = FlipCommandCfg(
    fixed_stage=2,  # S3: Low / Mid / High.
    no_trigger_prob=0.0,
    command_change_prob=1.0,
    trigger_delay_range_s=(1.0, 1.0),
    input_jitter_steps=0,
  )
  env = make_plain_env(num_envs=16, commands={"flip": cfg})
  flip = env.command_manager.get_term("flip")
  env.reset()
  _step_for(env, 2.0)
  assert flip.triggered.all()
  assert torch.isinf(flip.cmd_change_time).all()  # Every scheduled change happened.
  expected = flip.cmd_change_idx.clone()
  assert torch.equal(flip.cmd_idx, expected)
  # Changes after the trigger are ignored (EV-2).
  t = env.episode_length_buf.float() * env.step_dt
  flip.cmd_change_time[:] = t + 0.1
  flip.cmd_change_idx[:] = (expected + 1) % 3
  _step_for(env, 0.5)
  assert torch.equal(flip.cmd_idx, expected)
  scalars = torch.tensor(C.COMMAND_SCALARS)[expected]
  assert torch.allclose(flip.command[:, 0], scalars)


def test_interlock_tolerates_exploration_noise():
  """Standing with action noise like the training policy (std 0.14) must
  still pass the EV-6 interlock. Pure noise around the default pose, with no
  controller: the filtered check accepts ~70 % within 2 s, the old
  instantaneous per-joint check 0 %."""
  cfg = scripted_flip_cfg()
  env = make_plain_env(num_envs=16, commands={"flip": cfg})
  flip = env.command_manager.get_term("flip")
  torch.manual_seed(0)
  env.reset()
  for _ in range(int(2.0 / env.step_dt)):
    env.step(0.14 * torch.randn(env.num_envs, env.action_manager.total_action_dim))
  assert flip.triggered.float().mean() >= 0.5
