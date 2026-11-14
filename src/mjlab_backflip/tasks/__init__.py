"""Task registration (loaded by mjlab through the ``mjlab.tasks`` entry point)."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_backflip.tasks.backflip.backflip_env_cfg import make_backflip_env_cfg
from mjlab_backflip.tasks.backflip.rl_cfg import backflip_ppo_runner_cfg
from mjlab_backflip.tasks.velocity.go2_env_cfg import unitree_go2_flat_env_cfg
from mjlab_backflip.tasks.velocity.rl_cfg import unitree_go2_ppo_runner_cfg

# Sanity task: Go2 flat velocity tracking, ported from unitree_rl_mjlab.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go2",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# One backflip per episode, height command Low / Mid / High (see the spec).
register_mjlab_task(
  task_id="Mjlab-Backflip-Flat-Unitree-Go2",
  env_cfg=make_backflip_env_cfg(),
  play_env_cfg=make_backflip_env_cfg(play=True),
  rl_cfg=backflip_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
