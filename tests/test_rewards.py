"""Time-gated reward set (revised 2026-11-15): signs, windows, and that
flipping pays more than staying put after the trigger."""

import math

import pytest
import torch

import mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from test_terminations import _set_base_velocity
from mjlab_backflip.robots import GO2_SPEC
from mjlab_backflip.tasks.backflip import constants as C
from mjlab_backflip.tasks.backflip.mdp.rewards import reference_gravity

TASK = "Mjlab-Backflip-Flat-Unitree-Go2"
POSITIVE = {"flip_ang_vel", "flip_lin_vel_z", "flip_landing"}
WINDOWED = {"flip_ang_vel": (C.FLIP_START_S, C.FLIP_ROTATE_S),
            "flip_lin_vel_z": (C.FLIP_START_S, C.FLIP_LIFT_S)}
TRIGGER_AT = 0.5


def test_weight_signs():
  for name, w in C.REWARD_WEIGHTS.items():
    assert (w > 0) == (name in POSITIVE), name


class _FakeFlip:
  spec = GO2_SPEC

  def __init__(self, tau):
    self._tau = torch.tensor(tau)

  def time_since_trigger(self):
    return self._tau


def test_reference_attitude_turns_one_backflip():
  s, r = C.FLIP_START_S, C.FLIP_ROTATE_S
  taus = [-math.inf, 0.0, s + 0.25 * r, s + 0.5 * r, s + 0.75 * r, s + r, 2.0]
  g = reference_gravity(_FakeFlip(taus))
  up = torch.tensor([0.0, 0.0, -1.0])
  for i in (0, 1, 5, 6):
    assert torch.allclose(g[i], up, atol=1e-6)
  assert g[3, 2] == pytest.approx(1.0, abs=1e-6)  # Upside down halfway.
  # Same direction as the state machine's theta (angles wrap to (-pi, pi]).
  angles = GO2_SPEC.back_pitch_angle(g[2:5, 0], g[2:5, 2])
  ref = torch.tensor([0.5, 1.0, 1.5]) * math.pi
  assert torch.allclose(torch.cos(angles), torch.cos(ref), atol=1e-5)
  assert torch.allclose(torch.sin(angles), torch.sin(ref), atol=1e-5)


def _make_env():
  cfg = load_env_cfg(TASK, play=True)  # Training terminations, no DR / noise.
  cfg.scene.num_envs = 1
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  flip_cfg = cfg.commands["flip"]
  flip_cfg.fixed_command = C.MID
  flip_cfg.command_change_prob = 0.0
  flip_cfg.trigger_delay_range_s = (TRIGGER_AT, TRIGGER_AT)
  flip_cfg.input_jitter_steps = 0
  flip_cfg.interlock = False
  flip_cfg.fixed_stage = 0
  return ManagerBasedRlEnv(cfg=cfg, device="cpu")


@pytest.fixture(scope="module")
def env():
  return _make_env()


def _episode(env, throw: bool, trigger: bool = True):
  """One episode; returns (per-term totals, per-step (tau, raw values), done terms)."""
  flip = env.command_manager.get_term("flip")
  rm = env.reward_manager
  names = list(rm.active_terms)
  weights = torch.tensor([rm.get_term_cfg(n).weight for n in names])
  env.reset()
  if not trigger:
    flip.trigger_input_time[:] = math.inf
  thrown = landed = False
  steps, sums = [], torch.zeros(len(names))
  for _ in range(int(4.0 / env.step_dt)):
    tau = float(flip.time_since_trigger()[0])
    if throw and not thrown and tau >= C.FLIP_START_S:
      # Hand-thrown flip at the start of the window (the scripted legs
      # cannot jump high enough, see docs/notes.md).
      _set_base_velocity(env, 2.0, 1.6 * 2 * math.pi / C.FLIP_ROTATE_S)
      thrown = True
    if thrown and not landed and float(flip.theta[0]) >= 2 * math.pi - 1.0:
      landed = True  # Landing assist: upright just above the ground.
      robot = env.scene["robot"]
      state = robot.data.default_root_state.clone()
      state[:, :2] = robot.data.root_link_pos_w[:, :2]
      state[:, 2] += 0.03
      state[:, 7:] = 0.0
      robot.write_root_state_to_sim(state)
      robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    env.step(torch.zeros(1, 12))
    rate = rm._step_reward[0].clone()
    sums += rate * env.step_dt
    steps.append((float(flip.time_since_trigger()[0]) if not env.reset_buf[0] else tau,
                  rate / weights))
    if env.reset_buf[0]:
      tm = env.termination_manager
      done = {n for n in tm.active_terms if tm.get_term(n)[0]}
      return dict(zip(names, sums.tolist())), steps, done, names
  raise AssertionError("episode did not end")


def test_windows(env):
  _, steps, _, names = _episode(env, throw=True)
  idx = {n: i for i, n in enumerate(names)}
  for name, (start, length) in WINDOWED.items():
    active = [tau for tau, raw in steps if raw[idx[name]].abs() > 0]
    assert active, name
    assert min(active) >= start - 1e-6 and max(active) < start + length + 0.03, name
  # Feet are held down only before the flip; height only outside it.
  for tau, raw in steps:
    if tau >= C.FLIP_START_S + 0.03:
      assert raw[idx["feet_height_before"]] == 0.0, tau
    if 0.0 <= tau <= C.FLIP_START_S + C.FLIP_SETTLE_S:
      assert raw[idx["height_control"]] == 0.0, tau


def test_untriggered_episode_has_no_flip_reward(env):
  sums, _, done, _ = _episode(env, throw=False, trigger=False)
  assert sums["flip_ang_vel"] == 0.0 and sums["flip_lin_vel_z"] == 0.0
  assert sums["flip_landing"] == 0.0
  assert done == {"time_out"}


def test_flipping_pays_more_than_staying_put(env):
  """The loophole check: after the trigger, standing still must lose to a
  flip. The flip episode ends at trigger + FLIP_TAIL_S, without failure
  terminations."""
  still, _, done_still, _ = _episode(env, throw=False)
  flip, _, done_flip, _ = _episode(env, throw=True)
  assert done_still == {"flip_tail"}
  assert done_flip <= {"flip_tail", "success_tail"} and done_flip
  assert flip["flip_ang_vel"] > 5.0 and flip["flip_lin_vel_z"] > 0.0
  # The landing term pays only after a full turn: standing still earns ~0.
  assert flip["flip_landing"] > 3.0 and still["flip_landing"] < 1e-3
  assert sum(flip.values()) > sum(still.values()) + 10.0, (flip, still)
