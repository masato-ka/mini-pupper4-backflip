"""Lock the backflip sign convention (see scripts/check_signs.py)."""

import torch

from _sim_utils import make_plain_env
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from mjlab_backflip.robots import GO2_SPEC


def _nose_minus_tail(robot) -> float:
  ids = [robot.body_names.index(n) for n in ("FL_hip", "FR_hip", "RL_hip", "RR_hip")]
  z = robot.data.body_link_pos_w[0, ids, 2]
  return ((z[0] + z[1]) / 2 - (z[2] + z[3]) / 2).item()


def test_backflip_direction_signs():
  env = make_plain_env()
  robot = env.scene["robot"]
  gyro = env.scene["robot/imu_ang_vel"]
  env.reset()

  # Spin about the body y axis in the backflip direction.
  w_back = 2.0
  state = robot.data.default_root_state.clone()
  state[:, 2] = 1.0
  state[:, 3:7] = quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.zeros(1))
  state[:, 7:] = 0.0
  robot.write_root_state_to_sim(state)
  robot.write_root_link_velocity_b_to_sim(
    torch.tensor([[0.0, 0.0, 0.0, 0.0, GO2_SPEC.back_pitch_sign * w_back, 0.0]])
  )
  env.sim.forward()
  env.sim.sense()
  g0 = robot.data.projected_gravity_b[0]
  angle0 = GO2_SPEC.back_pitch_angle(g0[0].item(), g0[2].item())
  nose0 = _nose_minus_tail(robot)
  omega_back = GO2_SPEC.back_pitch_sign * gyro.data[0, 1].item()

  for _ in range(5):
    env.sim.step()
  env.sim.forward()
  g1 = robot.data.projected_gravity_b[0]
  angle1 = GO2_SPEC.back_pitch_angle(g1[0].item(), g1[2].item())

  assert omega_back > 0.0  # IMU gyro reports the backflip rate as positive.
  assert _nose_minus_tail(robot) > nose0  # The nose goes up first.
  assert angle1 > angle0  # The flip angle grows in the backflip direction.
