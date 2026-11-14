"""Measure the settled stand height of the robot under zero action (M1).

Reports two heights:
  - kinematic: base height with the default pose and the feet just touching
    the ground (the reset height; avoids the drop at reset).
  - settled: where the base ends up when the PD actuators hold the default
    pose against gravity with zero action (passive sag).

  uv run python scripts/measure_stand_height.py [--seconds 10]
"""

import argparse

import torch

from _sim_utils import make_plain_env

from mjlab_backflip.robots import GO2_SPEC

FOOT_RADIUS = 0.022  # Foot collision sphere radius in go2.xml.


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--seconds", type=float, default=10.0)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  env = make_plain_env(device=args.device)
  robot = env.scene["robot"]
  env.reset()
  act = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  n_steps = int(args.seconds / env.step_dt)
  report = {int(t / env.step_dt) for t in (0.5, 1, 2, 3, 5, 7.5, 10, 15, 20)}

  z0 = robot.data.root_link_pos_w[0, 2].item()
  print(f"reset z = {z0:.4f} m")
  # Kinematic height: base height at which the default-pose feet touch ground.
  foot_ids = [robot.site_names.index(n) for n in GO2_SPEC.foot_names]
  foot_z = robot.data.site_pos_w[0, foot_ids, 2]
  h_kin = z0 - (foot_z.min().item() - FOOT_RADIUS)
  print(f"kinematic stand height (default pose, feet on ground) = {h_kin:.4f} m")
  prev_z = z0
  for i in range(1, n_steps + 1):
    env.step(act)
    z = robot.data.root_link_pos_w[0, 2].item()
    if i in report or i == n_steps:
      dz = (z - prev_z) / env.step_dt
      print(f"t={i * env.step_dt:5.2f}s  z={z:.4f} m  dz/dt={dz * 1000:+.2f} mm/s")
    prev_z = z

  q_err = robot.data.joint_pos[0] - robot.data.default_joint_pos[0]
  tau = robot.data.actuator_force[0]
  g = robot.data.projected_gravity_b[0]
  print("joint names :", robot.joint_names)
  print("q - q_default:", [round(v, 3) for v in q_err.tolist()])
  print("actuator tau :", [round(v, 2) for v in tau.tolist()])
  print("gravity_b    :", [round(v, 3) for v in g.tolist()])
  print(f"settled stand height under zero action = {z:.4f} m")


if __name__ == "__main__":
  main()
