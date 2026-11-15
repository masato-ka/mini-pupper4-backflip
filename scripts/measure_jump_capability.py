"""Upper bound check for the Go2 jump: crouch, then extend every leg at once,
at the default PD gains and at x3.5 (~ Kp 70). Prints the peak rise and v_z.

  uv run python scripts/measure_jump_capability.py

Result (2026-11-15, CPU): rise 0.14 m / 1.49 m/s at the default gains and
0.15 m / 1.68 m/s at x3.5, far from the spec's dh* 0.25-0.45 m.
"""

import torch

from _sim_utils import make_plain_env
from check_state_machine import leg_action


def run(kp_scale, push):
  env = make_plain_env()
  robot = env.scene["robot"]
  m = env.sim.model
  ids = robot.indexing.ctrl_ids
  m.actuator_gainprm[..., ids, 0] *= kp_scale
  m.actuator_biasprm[..., ids, 1] *= kp_scale
  m.actuator_biasprm[..., ids, 2] *= max(1.0, kp_scale ** 0.5)
  env.reset()
  dt = env.step_dt
  h0 = None; hmax = 0; vzmax = 0; airborne = 0
  for k in range(int(2.0 / dt)):
    t = k * dt
    if t < 0.5: a = leg_action(robot, 0, 0)
    elif t < 1.0: a = leg_action(robot, 0.8, -0.9)   # crouch
    else: a = leg_action(robot, -push, push)          # extend
    env.step(a)
    z = float(robot.data.root_link_pos_w[0, 2]); vz = float(robot.data.root_link_lin_vel_w[0, 2])
    if t < 0.5: h0 = z
    if t >= 1.0:
      hmax = max(hmax, z); vzmax = max(vzmax, vz)
  print(f"kp x{kp_scale:<4} push {push:<4}: h_stand {h0:.3f}  peak z {hmax:.3f} (rise {hmax-h0:+.3f})  vz_max {vzmax:.2f} m/s")
  env.close()

if __name__ == "__main__":
  for kp in (1.0, 3.5):
    for push in (0.5, 1.0, 2.0, 4.0):
      run(kp, push)
