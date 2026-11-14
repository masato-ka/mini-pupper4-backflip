"""Roll out a trained policy headless, save an mp4 and print the flip outcome.

mjlab's ``play`` opens an interactive viewer after recording, which blocks a
Colab cell. This script only renders offscreen and exits.

  uv run python scripts/record_video.py --checkpoint logs/rsl_rl/go2_backflip/<run>/model_1000.pt \
      [--command mid] [--seconds 4] [--out flip.mp4] [--device cuda:0]
"""

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

if sys.platform.startswith("linux"):
  os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy as media
import torch

import mjlab  # noqa: F401  (loads the mjlab.tasks entry points)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_backflip.tasks.backflip import constants as C

TASK = "Mjlab-Backflip-Flat-Unitree-Go2"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--command", choices=C.COMMAND_NAMES, default="mid")
  parser.add_argument("--trigger-at", type=float, default=1.0, help="trigger input time [s]")
  parser.add_argument("--seconds", type=float, default=4.0)
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument("--out", default=None, help="mp4 path (default: next to the checkpoint)")
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()

  ckpt = Path(args.checkpoint)
  out = Path(args.out) if args.out else ckpt.parent / "videos" / f"{ckpt.stem}_{args.command}.mp4"
  out.parent.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(TASK, play=True)  # Final stage, no DR, no corruption.
  env_cfg.scene.num_envs = 1
  env_cfg.episode_length_s = args.seconds
  env_cfg.viewer.width, env_cfg.viewer.height = args.width, args.height
  flip_cfg = env_cfg.commands["flip"]
  flip_cfg.fixed_command = C.COMMAND_NAMES.index(args.command)
  flip_cfg.command_change_prob = 0.0
  flip_cfg.trigger_delay_range_s = (args.trigger_at, args.trigger_at)
  agent_cfg = load_rl_cfg(TASK)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=args.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  flip = env.command_manager.get_term("flip")

  obs = vec_env.get_observations()
  frames, log, prev = [], [], 0
  for step in range(int(args.seconds / env.step_dt)):
    with torch.no_grad():
      obs, _, dones, _ = vec_env.step(policy(obs))
    frame = env.render()
    if frame is not None:
      frames.append(frame)
    p = int(flip.phase[0])
    if p != prev and not bool(dones[0]):
      log.append(f"t={(step + 1) * env.step_dt:.2f}s  {C.PHASE_NAMES[prev]} -> {C.PHASE_NAMES[p]}")
      if p == C.LAND:
        log.append(
          f"           theta_td={flip.theta_td[0]:.2f} rad, f_flip={flip.f_flip[0]:.2f}, "
          f"H_max={flip.H_max[0]:.3f} m (target {flip.target_height[0]:.2f})"
        )
    if bool(dones[0]):
      tm = env.termination_manager
      fired = [n for n in tm.active_terms if tm.get_term(n)[0]]
      log.append(f"t={(step + 1) * env.step_dt:.2f}s  episode end: {fired}")
      break
    prev = p

  fps = round(1.0 / env.step_dt)
  media.write_video(str(out), frames, fps=fps)
  print("\n".join(log))
  print(f"[INFO] {len(frames)} frames -> {out}")


if __name__ == "__main__":
  main()
