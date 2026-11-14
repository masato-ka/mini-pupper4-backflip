from mjlab_backflip.tasks.backflip.mdp import rewards
from mjlab_backflip.tasks.backflip.mdp.curriculum import flip_curriculum
from mjlab_backflip.tasks.backflip.mdp.events import hold_default_joint_targets
from mjlab_backflip.tasks.backflip.mdp.flip_command import FlipCommand, FlipCommandCfg
from mjlab_backflip.tasks.backflip.mdp.observations import (
  attitude_estimate,
  base_ang_vel_b,
  base_lin_vel_w,
  dr_params,
  flip_command_obs,
  flip_state,
  foot_contact,
  foot_contact_force,
  foot_height,
  imu_reading,
  joint_torque,
)
from mjlab_backflip.tasks.backflip.mdp.terminations import (
  base_contact,
  base_too_low,
  forward_rotation,
  over_rotation,
  phase_timeout,
  success_tail_elapsed,
  under_rotated_landing,
  unstable_sim,
)
