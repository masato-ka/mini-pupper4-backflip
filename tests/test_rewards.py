"""Reward table (M4): signs, gating, and the sparse terms firing once."""

import math

import pytest
import torch

import mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from test_terminations import _thrower
from mjlab_backflip.tasks.backflip import constants as C

TASK = "Mjlab-Backflip-Flat-Unitree-Go2"

POSITIVE = {
  "crouch_depth", "takeoff_vz", "pitch_kick", "apex_bonus", "flip_progress",
  "land_prep", "upright", "land_height", "feet_all_contact", "success_bonus", "alive",
}
STAND_TERMS = {"stand_height", "stand_pose", "stand_still", "feet_distance"}
AIR_TERMS = {"apex_bonus", "apex_overshoot", "flip_progress", "wrong_direction", "tuck", "land_prep"}
LAND_TERMS = {
  "upright", "land_height", "feet_all_contact", "land_damp", "land_effort", "land_action_rate",
}


def test_weight_signs_match_spec():
  for name, w in C.REWARD_WEIGHTS.items():
    if name in ("tuck", "land_prep"):
      assert w == 0.0  # Optional terms start disabled.
    elif name in POSITIVE:
      assert w > 0, name
    else:
      assert w < 0, name
  assert C.TUCK_WEIGHT_SPEC < 0 < C.LAND_PREP_WEIGHT_SPEC


@pytest.fixture(scope="module")
def rollout():
  """Thrown flip with a landing assist; records the per-step reward rates."""
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  flip_cfg = cfg.commands["flip"]
  flip_cfg.fixed_command = C.MID
  flip_cfg.command_change_prob = 0.0
  flip_cfg.trigger_delay_range_s = (0.5, 0.5)
  flip_cfg.input_jitter_steps = 0
  flip_cfg.fixed_stage = 0
  # Enable the optional terms so their signs are checked too.
  cfg.rewards["tuck"].weight = C.TUCK_WEIGHT_SPEC
  cfg.rewards["land_prep"].weight = C.LAND_PREP_WEIGHT_SPEC
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  rm = env.reward_manager
  names = list(rm.active_terms)
  weights = torch.tensor([rm.get_term_cfg(n).weight for n in names])
  flip = env.command_manager.get_term("flip")

  dh = C.JUMP_HEIGHTS[C.MID]
  throw = _thrower(vz=C.takeoff_speed(dh), w_back=1.6 * C.flip_rate(dh))
  assisted = []
  env.reset()
  prev = now = int(flip.phase[0])
  steps = []  # (phase used for the reward, raw values per term)
  sums = torch.zeros(len(names))
  for _ in range(int(5.0 / env.step_dt)):
    if not assisted and now == C.AIR and flip.theta[0] >= 2 * math.pi - 1.0:
      assisted.append(True)
      robot = env.scene["robot"]
      state = robot.data.default_root_state.clone()
      state[:, :2] = robot.data.root_link_pos_w[:, :2]
      state[:, 2] += 0.03
      state[:, 7:] = 0.0
      robot.write_root_state_to_sim(state)
      robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    env.step(throw(env, flip, prev))
    done = bool(env.reset_buf[0])
    rate = rm._step_reward[0].clone()
    sums += rate * env.step_dt
    # The phase the rewards saw: after a reset flip.phase is already 0.
    steps.append((now if done else int(flip.phase[0]), rate / torch.where(weights == 0, 1.0, weights)))
    prev, now = now, int(flip.phase[0])
    if done:
      break
  return names, weights, steps, dict(zip(names, sums.tolist()))


def test_raw_values_are_non_negative(rollout):
  names, weights, steps, _ = rollout
  for phase, raw in steps:
    for name, w, v in zip(names, weights, raw):
      if w != 0:
        assert v >= -1e-6, (name, C.PHASE_NAMES[phase], float(v))


def test_phase_gating(rollout):
  names, _, steps, _ = rollout
  idx = {n: i for i, n in enumerate(names)}
  seen = set()
  for phase, raw in steps:
    seen.add(phase)
    active = {n for n in names if raw[idx[n]].abs() > 0}
    if phase != C.STAND:
      assert not (active & STAND_TERMS), (C.PHASE_NAMES[phase], active & STAND_TERMS)
    if phase != C.AIR:
      assert not (active & AIR_TERMS), (C.PHASE_NAMES[phase], active & AIR_TERMS)
    if phase not in (C.LAND, C.DONE):
      assert not (active & LAND_TERMS), (C.PHASE_NAMES[phase], active & LAND_TERMS)
  assert seen == set(range(C.NUM_PHASES))


def test_sparse_terms_fire_once(rollout):
  names, weights, steps, sums = rollout
  idx = {n: i for i, n in enumerate(names)}
  for name in ("apex_bonus", "success_bonus"):
    fired = [p for p, raw in steps if raw[idx[name]] > 0]
    assert len(fired) == 1, (name, fired)
  # Episode totals: success is the full weight, the apex at most the weight.
  assert sums["success_bonus"] == pytest.approx(C.REWARD_WEIGHTS["success_bonus"], rel=1e-4)
  assert 0 < sums["apex_bonus"] <= C.REWARD_WEIGHTS["apex_bonus"] + 1e-4
  # The success tail is a truncation: no failure penalty, alive all along.
  assert sums["terminal_penalty"] == 0.0
  assert sums["alive"] == pytest.approx(len(steps) * C.REWARD_WEIGHTS["alive"] * 0.02, rel=1e-4)
