"""Evaluate a checkpoint with the mean action (no exploration noise).

Training metrics come from noisy actions: EV-9 success needs the body still
for 0.2-0.4 s, which the exploration noise never allows, so
Metrics/flip/rate/success reads 0 even when the mean policy succeeds (Colab
run 2026-11-15_01-50-56: 0 % in training, 64 % here). Use this script to
judge a policy against EV-9.

  uv run python scripts/evaluate_policy.py --checkpoint logs/rsl_rl/go2_backflip/<run>/model_1000.pt \
      [--level 2] [--command mid] [--num-envs 64] [--noise-std 0] [--interlock] [--device cuda:0]

The EV-6 interlock is off by default, as in training S1-S3. With
--interlock the trigger waits for a still stand, as on the real robot and in
S4; a policy trained without it may never get the trigger accepted (model_750
of that run: 0-2 % within the episode).

Prints the success / landed / rotation rates, how often each EV-9 condition
holds during Land, and the median trajectory against the time since the
trigger.
"""

import argparse
import math
import os
import sys
from dataclasses import asdict

if sys.platform.startswith("linux"):
  os.environ.setdefault("MUJOCO_GL", "egl")

import torch

import mjlab  # noqa: F401  (loads the mjlab.tasks entry points)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_backflip.tasks.backflip import constants as C

TASK = "Mjlab-Backflip-Flat-Unitree-Go2"
TAU_ROWS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 1.95)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--level", type=int, default=len(C.LEVELS) - 1, help="0..2 (success judgement)")
  parser.add_argument("--command", choices=C.COMMAND_NAMES, default="mid")
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--noise-std", type=float, default=0.0, help="action noise added to the mean")
  parser.add_argument("--interlock", action="store_true", help="EV-6: wait for a still stand")
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()

  n = args.num_envs
  cfg = load_env_cfg(TASK, play=True)  # No DR, no observation noise.
  cfg.scene.num_envs = n
  flip_cfg = cfg.commands["flip"]
  flip_cfg.fixed_command = C.COMMAND_NAMES.index(args.command)
  flip_cfg.command_change_prob = 0.0
  flip_cfg.fixed_level = args.level
  flip_cfg.interlock = args.interlock
  agent_cfg = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(vec_env, asdict(agent_cfg), device=args.device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  flip = env.command_manager.get_term("flip")
  robot = env.scene["robot"]
  lvl = C.LEVELS[args.level]

  steps = int(env.max_episode_length) + 1
  keys = ("phase", "theta", "h", "up", "v", "w", "nc", "tau")
  log = {k: torch.full((steps, n), math.nan, device=args.device) for k in keys}
  outcome = {k: torch.zeros(n, dtype=torch.bool, device=args.device) for k in ("success", "landed", "trig")}
  finished = torch.zeros(n, dtype=torch.bool, device=args.device)
  obs = vec_env.get_observations()
  for t in range(steps):
    with torch.no_grad():
      action = policy(obs)
      if args.noise_std > 0:
        action = action + args.noise_std * torch.randn_like(action)
    # Latch the outcome before the step: a finished env is reset inside it.
    outcome["success"] |= flip.success & ~finished
    outcome["landed"] |= flip.landed & ~finished
    outcome["trig"] |= flip.triggered & ~finished
    obs, _, dones, _ = vec_env.step(action)
    d = robot.data
    row = {
      "phase": flip.phase.float(), "theta": flip.theta, "h": flip.base_height(),
      "up": -d.projected_gravity_b[:, 2], "v": d.root_link_lin_vel_w.norm(dim=-1),
      "w": d.root_link_ang_vel_b.norm(dim=-1), "nc": flip.foot_contact().sum(-1).float(),
      "tau": flip.time_since_trigger(),
    }
    live = ~finished & ~dones.bool()
    for k, v in row.items():
      log[k][t] = torch.where(live, v, math.nan)
    finished |= dones.bool()
    if finished.all():
      break

  trig = outcome["trig"]
  print(
    f"\n{args.checkpoint}  level L{args.level}, {args.command}, noise std {args.noise_std}, "
    f"interlock {'on' if args.interlock else 'off'}"
  )
  print(f"triggered            {trig.float().mean():.2f}")
  if not trig.any():
    return
  reached = (log["phase"] >= C.LAND).any(0) & trig
  print(f"reached Land         {reached.float().sum() / trig.sum():.2f}")
  print(f"landed               {(outcome['landed'] & trig).float().sum() / trig.sum():.2f}")
  print(f"success (EV-9)       {(outcome['success'] & trig).float().sum() / trig.sum():.2f}")

  land = log["phase"] == C.LAND
  total = max(1, int(land.sum()))
  conds = {
    "all 4 feet": log["nc"] == 4,
    f"theta >= 2pi - {lvl.success_theta_margin}": log["theta"] >= 2 * math.pi - lvl.success_theta_margin,
    "upright": log["up"] >= C.UPRIGHT_MIN_COS,
    f"|v| < {C.SETTLED_MAX_LIN_VEL}": log["v"] < C.SETTLED_MAX_LIN_VEL,
    f"|w| < {C.SETTLED_MAX_ANG_VEL}": log["w"] < C.SETTLED_MAX_ANG_VEL,
  }
  print(f"\nshare of Land steps meeting each EV-9 condition (hold {lvl.success_hold_s} s needed):")
  every = torch.ones_like(land)
  for name, ok in conds.items():
    every &= ok
    print(f"  {name:22s} {int((ok & land).sum()) / total:.2f}")
  print(f"  {'all of them':22s} {int((every & land).sum()) / total:.2f}")

  print("\nmedian over envs, by time since the trigger:")
  print("  tau  phase  theta     h     up    |v|    |w|  feet")
  for tau in TAU_ROWS:
    m = (log["tau"] >= tau) & (log["tau"] < tau + 0.02)
    if not m.any():
      continue

    def med(k):
      x = log[k][m]
      return float(x[~torch.isnan(x)].median())

    print(
      f" {tau:4.2f}  {med('phase'):5.1f}  {med('theta'):5.2f}  {med('h'):5.2f}  {med('up'):5.2f}"
      f"  {med('v'):5.2f}  {med('w'):5.2f}  {med('nc'):4.1f}"
    )


if __name__ == "__main__":
  main()
