"""Open-loop check of the flip state machine, no learning (M2).

A scripted controller crouches after the trigger. Go2 cannot flip open-loop,
so when Launch starts the script hands the base the ballistic velocity of the
Mid command (v*, omega*) — the "hand rotation" of the plan — and holds the
default pose. With ``--assist-landing`` it also sets the base upright at
touchdown so the T5 / success path is exercised deterministically.

Checks: phases advance 0 -> 5 in order, never go back, the trigger is
accepted once, and theta / h_to / f_flip are latched sensibly.

  uv run python scripts/check_state_machine.py [--assist-landing]
"""

import argparse
import math

import torch

from _sim_utils import make_plain_env

from mjlab_backflip.tasks.backflip import constants as C
from mjlab_backflip.tasks.backflip.mdp import FlipCommandCfg


def leg_action(robot, thigh: float, calf: float) -> torch.Tensor:
  """Action vector (actuator order) with the same offset on every leg."""
  a = torch.zeros(1, len(robot.actuator_names))
  for i, name in enumerate(robot.actuator_names):
    if "thigh" in name:
      a[0, i] = thigh
    elif "calf" in name:
      a[0, i] = calf
  return a


def scripted_flip_cfg() -> FlipCommandCfg:
  """Deterministic Mid flip triggered at 0.5 s."""
  return FlipCommandCfg(
    fixed_command=C.MID,
    no_trigger_prob=0.0,
    command_change_prob=0.0,
    trigger_delay_range_s=(0.5, 0.5),
    input_jitter_steps=0,
  )


def run_scripted_flip(env, assist_landing: bool, seconds: float = 4.0):
  """Run the scripted flip in env 0. Returns (phase per step, event log)."""
  robot = env.scene["robot"]
  flip = env.command_manager.get_term("flip")
  env.reset()

  stand = leg_action(robot, 0.0, 0.0)
  # Offsets keep calf = -2 * thigh so the feet stay under the hips.
  # Target = default + 0.5 * action (default thigh 0.9, calf -1.8).
  crouch = leg_action(robot, 0.6, -1.2)  # thigh 1.2, calf -2.4
  dh = C.JUMP_HEIGHTS[C.MID]

  phases, events = [], []
  prev = C.STAND
  for _ in range(int(seconds / env.step_dt)):
    env.step(crouch if int(flip.phase[0]) == C.CROUCH else stand)
    p = int(flip.phase[0])
    t = env.episode_length_buf[0].item() * env.step_dt

    if p == C.CROUCH and prev == C.STAND:
      events.append(f"t={t:.2f} T1 trigger accepted (cmd={C.COMMAND_NAMES[int(flip.cmd_idx[0])]})")
    if p == C.LAUNCH and prev == C.CROUCH:
      events.append(f"t={t:.2f} T2 crouched: h={flip.base_height()[0]:.3f} < {flip.crouch_height[0]:.3f}")
      # Hand rotation: Mid ballistic takeoff in the backflip direction.
      vel_b = torch.tensor([[0.0, 0.0, C.takeoff_speed(dh), 0.0, 0.0, 0.0]])
      vel_b[0, 4] = flip.spec.back_pitch_sign * C.flip_rate(dh)
      robot.write_root_link_velocity_b_to_sim(vel_b)
      events.append(f"t={t:.2f} hand rotation applied (v*={C.takeoff_speed(dh):.2f}, w*={C.flip_rate(dh):.1f})")
    if p == C.AIR and prev == C.LAUNCH:
      events.append(f"t={t:.2f} T3 takeoff: h_to={flip.h_to[0]:.3f}")
    if flip.apex_now[0]:
      events.append(f"t={t:.2f} apex: H={flip.H_max[0]:.3f} m (target {flip.target_height[0]:.2f})")
    if p == C.LAND and prev == C.AIR:
      events.append(
        f"t={t:.2f} T4 touchdown: theta_td={flip.theta_td[0]:.2f} rad "
        f"({math.degrees(flip.theta_td[0]):.0f} deg), f_flip={flip.f_flip[0]:.2f}, "
        f"H_max={flip.H_max[0]:.3f} m"
      )
      if assist_landing:
        state = robot.data.default_root_state.clone()
        state[:, :2] = robot.data.root_link_pos_w[:, :2]
        state[:, 7:] = 0.0
        robot.write_root_state_to_sim(state)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        events.append(f"t={t:.2f} landing assist: base set upright at h_nom")
    if p == C.DONE and prev == C.LAND:
      events.append(f"t={t:.2f} T5 success: theta={flip.theta[0]:.2f} rad")
    phases.append(p)
    prev = p
  return phases, events


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--assist-landing", action="store_true")
  parser.add_argument("--seconds", type=float, default=4.0)
  args = parser.parse_args()

  env = make_plain_env(commands={"flip": scripted_flip_cfg()})
  flip = env.command_manager.get_term("flip")
  phases, events = run_scripted_flip(env, args.assist_landing, args.seconds)

  print("\n".join(events))
  diffs = [b - a for a, b in zip(phases, phases[1:])]
  changes = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
  print(f"phase sequence (changes only): {changes}")
  print(f"monotonic: {all(d >= 0 for d in diffs)}, at most one transition per step: {all(d <= 1 for d in diffs)}")
  print(
    f"final: phase={C.PHASE_NAMES[phases[-1]]}, triggered={bool(flip.triggered[0])}, "
    f"success={bool(flip.success[0])}, theta={flip.theta[0]:.2f}, "
    f"theta range=[{flip.theta_min[0]:.2f}, {flip.theta_max[0]:.2f}]"
  )
  print(f"command obs: {[round(v, 3) for v in flip.command[0].tolist()]}")


if __name__ == "__main__":
  main()
