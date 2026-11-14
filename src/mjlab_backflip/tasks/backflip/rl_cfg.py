"""PPO configuration for the backflip task (Go2 velocity runner settings)."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_backflip.tasks.velocity.rl_cfg import unitree_go2_ppo_runner_cfg


def backflip_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.experiment_name = "go2_backflip"
  cfg.wandb_project = "mini-pupper4-backflip"
  cfg.max_iterations = 30_000
  cfg.save_interval = 250
  return cfg
