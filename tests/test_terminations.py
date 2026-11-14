"""Each failure / truncation fires for the scenario it is meant for (M3)."""

import math

import pytest
import torch

import mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from check_state_machine import leg_action
from mjlab_backflip.tasks.backflip import constants as C

TASK = "Mjlab-Backflip-Flat-Unitree-Go2"


@pytest.fixture(scope="module")
def env():
  cfg = load_env_cfg(TASK, play=True)  # No DR, no corruption.
  cfg.scene.num_envs = 1
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  flip = cfg.commands["flip"]
  flip.fixed_command = C.MID
  flip.command_change_prob = 0.0
  flip.trigger_delay_range_s = (0.5, 0.5)
  flip.input_jitter_steps = 0
  flip.fixed_stage = None
  return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def _run(env, level, controller, max_s=5.0):
  """Step until env 0 is done.

  controller(env, flip, prev) -> action, where prev is the phase before the
  last step (flip.phase is the current one).
  """
  flip = env.command_manager.get_term("flip")
  flip.level = level
  env.reset()
  prev = now = int(flip.phase[0])
  for _ in range(int(max_s / env.step_dt)):
    env.step(controller(env, flip, prev))
    prev, now = now, int(flip.phase[0])
    if env.reset_buf[0]:
      tm = env.termination_manager
      return {n for n in tm.active_terms if tm.get_term(n)[0]}
  return set()


def _stand(env, flip, prev):
  return torch.zeros(1, 12)


def _set_base_velocity(env, vz, w_back):
  robot = env.scene["robot"]
  flip = env.command_manager.get_term("flip")
  vel_b = torch.tensor([[0.0, 0.0, vz, 0.0, flip.spec.back_pitch_sign * w_back, 0.0]])
  robot.write_root_link_velocity_b_to_sim(vel_b)


def _thrower(vz, w_back, crouch=True):  # noqa: D103
  """Crouch after the trigger, then throw the base at the start of Launch."""

  def ctrl(env, flip, prev):
    robot = env.scene["robot"]
    p = int(flip.phase[0])
    if p == C.LAUNCH and prev == C.CROUCH:
      _set_base_velocity(env, vz, w_back)
    if crouch and p == C.CROUCH:
      return leg_action(robot, 0.6, -1.2)
    return torch.zeros(1, 12)

  return ctrl


def test_base_contact_on_back(env):
  def ctrl(env, flip, prev):
    if env.episode_length_buf[0] == 1:
      robot = env.scene["robot"]
      state = robot.data.default_root_state.clone()
      state[:, 2] = 0.5
      state[:, 3:7] = quat_from_euler_xyz(torch.tensor([math.pi]), torch.zeros(1), torch.zeros(1))
      robot.write_root_state_to_sim(state)
      # Teleporting jumps the pitch angle by pi; restart the theta reference.
      flip._needs_init[0] = True
    return torch.zeros(1, 12)

  assert "base_contact" in _run(env, 2, ctrl)


def test_forward_rotation(env):
  def ctrl(env, flip, prev):
    if env.episode_length_buf[0] == 5:
      _set_base_velocity(env, 4.0, -12.0)
    return torch.zeros(1, 12)

  assert "forward_rotation" in _run(env, 2, ctrl)


def test_over_rotation(env):
  assert "over_rotation" in _run(env, 2, _thrower(vz=4.0, w_back=25.0))


def test_crouch_timeout(env):
  # Trigger accepted but the robot never crouches below rho_c * h_nom.
  assert "phase_timeout" in _run(env, 2, _stand)


def test_under_rotated_landing(env):
  done = _run(env, 2, _thrower(vz=2.6, w_back=0.0))
  assert "under_rotated_landing" in done


def test_under_rotated_landing_disabled_at_l0(env):
  done = _run(env, 0, _thrower(vz=2.6, w_back=0.0), max_s=2.0)
  assert "under_rotated_landing" not in done


def test_success_tail_is_a_time_out(env):
  dh = C.JUMP_HEIGHTS[C.MID]
  # The thrown body with extended legs spins slower than the given rate and
  # reaches only ~270 deg at Mid omega*; throw faster so it passes 2 pi - 1.
  base = _thrower(vz=C.takeoff_speed(dh), w_back=1.6 * C.flip_rate(dh))

  assisted = []

  def ctrl(env, flip, prev):
    # Landing assist (once): when the rotation passes the L0 success angle in
    # the air, put the base upright just above the ground.
    if not assisted and int(flip.phase[0]) == C.AIR and flip.theta[0] >= 2 * math.pi - 1.0:
      assisted.append(True)
      robot = env.scene["robot"]
      state = robot.data.default_root_state.clone()
      state[:, :2] = robot.data.root_link_pos_w[:, :2]
      state[:, 2] += 0.03
      state[:, 7:] = 0.0
      robot.write_root_state_to_sim(state)
      robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    return base(env, flip, prev)

  done = _run(env, 0, ctrl)
  assert done == {"success_tail"}
  assert env.termination_manager.get_term_cfg("success_tail").time_out


def test_failures_are_terminal_and_truncations_are_time_outs(env):
  tm = env.termination_manager
  truncations = {n for n in tm.active_terms if tm.get_term_cfg(n).time_out}
  assert truncations == {"time_out", "success_tail"}


def test_no_hop_after_auto_reset(env):
  """The actuator delay buffer starts from the default pose after a reset."""
  robot = env.scene["robot"]
  flip = env.command_manager.get_term("flip")
  flip.level = 2
  env.reset()
  # Drive the joint targets far from the default, then force a reset.
  bent = torch.ones(1, 12)
  for _ in range(10):
    env.step(bent)
  ctrl_log = []
  orig = env.sim.step

  def logged_step():
    ctrl_log.append(env.sim.data.ctrl[0].clone())
    orig()

  env.episode_length_buf[:] = env.max_episode_length  # TM-1 on the next step.
  env.step(bent)
  assert env.reset_buf[0]
  env.sim.step = logged_step
  try:
    env.step(torch.zeros(1, 12))
  finally:
    env.sim.step = orig
  default_target = robot.data.default_joint_pos[0] - robot.data.encoder_bias[0]
  # ctrl is in actuator order; compare as sets of values per substep.
  for ctrl in ctrl_log:
    assert torch.allclose(ctrl.sort().values, default_target.sort().values, atol=1e-5)
