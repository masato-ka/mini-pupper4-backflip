"""Fix the sign conventions used by the flip state machine (M1).

Checks, with the robot held in the air:
  1. Which pitch direction is "nose up" (the first half of a backflip).
  2. The sign of phi = atan2(g_x, -g_z) from the body-frame gravity.
  3. The sign of the body-frame pitch rate (and the IMU gyro) during a
     nose-up rotation, i.e. how to compute omega_back.

  uv run python scripts/check_signs.py
"""

import math

import torch

from _sim_utils import make_plain_env
from mjlab.utils.lab_api.math import quat_from_euler_xyz


def front_minus_rear_z(robot) -> float:
  ids = [robot.body_names.index(n) for n in ("FL_hip", "FR_hip", "RL_hip", "RR_hip")]
  z = robot.data.body_link_pos_w[0, ids, 2]
  return ((z[0] + z[1]) / 2 - (z[2] + z[3]) / 2).item()


def phi_of(g: torch.Tensor) -> float:
  return math.atan2(g[0].item(), -g[2].item())


def main() -> None:
  env = make_plain_env()
  robot = env.scene["robot"]
  gyro = env.scene["robot/imu_ang_vel"]
  env.reset()

  print("== 1-2. static pitch: euler pitch -> nose height, phi ==")
  for pitch in (+0.3, -0.3):
    state = robot.data.default_root_state.clone()
    state[:, 2] = 1.0
    state[:, 3:7] = quat_from_euler_xyz(
      torch.zeros(1), torch.full((1,), pitch), torch.zeros(1)
    )
    state[:, 7:] = 0.0
    robot.write_root_state_to_sim(state)
    env.sim.forward()
    g = robot.data.projected_gravity_b[0]
    print(
      f"euler pitch {pitch:+.1f}: front-rear z = {front_minus_rear_z(robot):+.3f} m, "
      f"g_b = {[round(v, 3) for v in g.tolist()]}, phi = {phi_of(g):+.3f}"
    )

  print("== 3. body pitch rate +w_y (body frame) -> motion, gyro ==")
  for wy in (+2.0, -2.0):
    state = robot.data.default_root_state.clone()
    state[:, 2] = 1.0
    state[:, 7:] = 0.0
    robot.write_root_state_to_sim(state)
    robot.write_root_link_velocity_b_to_sim(
      torch.tensor([[0.0, 0.0, 0.0, 0.0, wy, 0.0]])
    )
    env.sim.forward()
    env.sim.sense()
    gyro_y = gyro.data[0, 1].item()
    d0 = front_minus_rear_z(robot)
    phi0 = phi_of(robot.data.projected_gravity_b[0])
    for _ in range(5):  # 5 physics substeps = 25 ms
      env.sim.step()
    env.sim.forward()
    d1 = front_minus_rear_z(robot)
    phi1 = phi_of(robot.data.projected_gravity_b[0])
    print(
      f"w_y {wy:+.1f}: gyro_y = {gyro_y:+.3f}, "
      f"nose {'UP' if d1 > d0 else 'DOWN'} ({d1 - d0:+.4f} m), dphi = {phi1 - phi0:+.4f}"
    )


if __name__ == "__main__":
  main()
