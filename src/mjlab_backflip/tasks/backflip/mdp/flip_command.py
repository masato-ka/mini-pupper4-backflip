"""Flip command: jump-height command, trigger, and the 5-phase state machine.

One flip per episode (no re-trigger). Phases only move forward:
  Stand -T1-> Crouch -T2-> Launch -T3-> Air -T4-> Land -T5-> Done.

State update order. mjlab runs terminations and rewards *before*
``command_manager.compute``. If the state machine only advanced in
``_update_command``, rewards and terminations would see the previous step's
phase. So ``update_state()`` is idempotent per env step (guarded by
``env.common_step_counter``) and is called from the first termination term as
well as from ``_update_command``; whichever comes first does the work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from mjlab_backflip.robots import GO2_SPEC, RobotSpec
from mjlab_backflip.tasks.backflip import constants as C

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(x), torch.cos(x))


class FlipCommand(CommandTerm):
  cfg: FlipCommandCfg

  def __init__(self, cfg: FlipCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.spec: RobotSpec = cfg.robot_spec
    n, dev = self.num_envs, self.device

    def zeros(dtype=torch.float):
      return torch.zeros(n, device=dev, dtype=dtype)

    # Curriculum-controlled (M5 writes these; fixed_* override).
    self.stage = cfg.fixed_stage if cfg.fixed_stage is not None else 0
    self.level = cfg.fixed_level if cfg.fixed_level is not None else C.STAGES[self.stage].level
    # EV-6 interlock on/off. The curriculum turns it off in S1-S3 (see
    # curriculum.py); play / evaluation keep the cfg value (on).
    self.interlock = cfg.interlock

    # Lookup tables indexed by command id.
    self._cmd_scalar = torch.tensor(C.COMMAND_SCALARS, device=dev)
    self._dh = torch.tensor(C.JUMP_HEIGHTS, device=dev)
    self._v_star = torch.tensor([C.takeoff_speed(h) for h in C.JUMP_HEIGHTS], device=dev)
    self._tf_star = torch.tensor([C.flight_time(h) for h in C.JUMP_HEIGHTS], device=dev)
    self._w_star = torch.tensor([C.flip_rate(h) for h in C.JUMP_HEIGHTS], device=dev)
    self._rho_c = torch.tensor(C.CROUCH_RATIOS, device=dev)

    # Command input (EV-1, EV-7).
    self.cmd_idx = zeros(torch.long)  # Current input; latched at trigger.
    self.cmd_change_time = zeros()  # inf = no change.
    self.cmd_change_idx = zeros(torch.long)
    # Trigger input (EV-3, EV-4, EV-8).
    self.trigger_input_time = zeros()  # inf = control episode, never fires.
    self.stable_time = zeros()  # EV-6 interlock timer.
    self.filt_ang_vel = zeros()  # EV-6 filtered base angular speed.
    self.filt_joint_vel = zeros()  # EV-6 filtered mean |joint velocity|.
    self.triggered = zeros(torch.bool)  # Latched, never cleared (EV-5).
    self.trigger_time = zeros()
    # Phase state.
    self.phase = zeros(torch.long)
    self.phase_time = zeros()
    self.max_phase = zeros(torch.long)
    self.theta = zeros()  # Accumulated backflip pitch [rad].
    self.theta_max = zeros()
    self.theta_min = zeros()
    self._angle_prev = zeros()
    self._angle_prev_ok = zeros(torch.bool)
    self.air_cnt = zeros(torch.long)  # Consecutive no-contact steps.
    self.air_steps = zeros(torch.long)  # Steps since entering Air.
    self.land_hold = zeros()
    self.h_to = zeros()  # Base height at liftoff (first no-contact step), latched at T3.
    self._h_liftoff = zeros()
    self.H_max = zeros()  # Max rise above h_to in Air.
    self.theta_td = zeros()  # Theta latched at T4.
    self.f_flip = zeros()  # clip(theta_td / (2 pi - 0.5), 0, 1), latched at T4.
    self.success = zeros(torch.bool)
    self.success_time = zeros()
    # One-step events (true only on the step they happen).
    self.entered = torch.zeros(n, C.NUM_PHASES, device=dev, dtype=torch.bool)
    self.apex_now = zeros(torch.bool)
    self.apex_done = zeros(torch.bool)  # The apex fires once per episode.
    self.success_now = zeros(torch.bool)
    self._vz_prev = zeros()
    self._needs_init = torch.ones(n, device=dev, dtype=torch.bool)
    self._last_update_step = -1

    # Per-episode stats, logged as Metrics/flip/<name> by CommandTerm.reset.
    for name in ("triggered", "success", "max_phase", "theta_td", "H_max"):
      self.metrics[name] = zeros()
    # EV-14 trackers, logged per finished episode in reset().
    self.phase_dur = torch.zeros(n, C.NUM_PHASES, device=dev)
    self.torque_peak = zeros()
    self.joint_vel_peak = zeros()
    self.land_acc_peak = zeros()
    self.start_xy = torch.zeros(n, 2, device=dev)
    self.start_yaw = zeros()
    # Pose at the last state update. reset() runs after the scene has been
    # reset, so the episode's final pose must be kept here.
    self.last_xy = torch.zeros(n, 2, device=dev)
    self.last_yaw = zeros()

  ##
  # Derived quantities (read by observations, rewards, terminations).
  ##

  @property
  def command(self) -> torch.Tensor:
    """Actor input: [command scalar, trigger flag, time since trigger / 2 s]."""
    t_since = torch.where(self.triggered, self._episode_time() - self.trigger_time, 0.0)
    return torch.stack(
      (
        self._cmd_scalar[self.cmd_idx],
        self.triggered.float(),
        (t_since / C.TRIGGER_TIME_NORM_S).clamp(max=1.0),
      ),
      dim=-1,
    )

  @property
  def target_height(self) -> torch.Tensor:
    return self._dh[self.cmd_idx]

  @property
  def v_star(self) -> torch.Tensor:
    return self._v_star[self.cmd_idx]

  @property
  def tf_star(self) -> torch.Tensor:
    return self._tf_star[self.cmd_idx]

  @property
  def omega_star(self) -> torch.Tensor:
    return self._w_star[self.cmd_idx]

  @property
  def crouch_height(self) -> torch.Tensor:
    return self._rho_c[self.cmd_idx] * self.spec.h_nom

  @property
  def level_cfg(self) -> C.LevelCfg:
    return C.LEVELS[self.level]

  @property
  def allowed_commands(self) -> tuple[int, ...]:
    return C.STAGES[self.stage].commands

  def base_height(self) -> torch.Tensor:
    return self.robot.data.root_link_pos_w[:, 2] - self._env.scene.env_origins[:, 2]

  def omega_back(self) -> torch.Tensor:
    """Body pitch rate, positive in the backflip direction."""
    return self.spec.back_pitch_sign * self.robot.data.root_link_ang_vel_b[:, 1]

  def foot_contact(self) -> torch.Tensor:
    sensor: ContactSensor = self._env.scene[self.cfg.contact_sensor_name]
    assert sensor.data.found is not None
    return sensor.data.found > 0

  def set_stage(self, stage: int, level: int | None = None) -> None:
    """Curriculum hook. Takes effect for commands drawn after the call."""
    if self.cfg.fixed_stage is None:
      self.stage = stage
    if self.cfg.fixed_level is None:
      self.level = C.STAGES[self.stage].level if level is None else level

  ##
  # State machine.
  ##

  def update_state(self) -> None:
    step = self._env.common_step_counter
    if step == self._last_update_step:
      return
    self._last_update_step = step
    dt = self._env.step_dt
    t = self._episode_time()
    data = self.robot.data
    lvl = self.level_cfg

    self.entered.zero_()
    self.apex_now.zero_()
    self.success_now.zero_()

    # Rotation progress theta. Wrapped increments of the gravity pitch angle
    # while the x-z projection of gravity is well defined; when the body rolls
    # towards 90 deg that angle is singular (g_x, g_z -> 0), so fall back to
    # integrating the body pitch rate.
    g = data.projected_gravity_b
    angle = self.spec.back_pitch_angle(g[:, 0], g[:, 2])
    init = self._needs_init
    proj_ok = torch.hypot(g[:, 0], g[:, 2]) >= C.THETA_MIN_PROJECTION
    d_angle = wrap_to_pi(angle - self._angle_prev)
    d_gyro = self.omega_back() * dt
    d_theta = torch.where(proj_ok & self._angle_prev_ok, d_angle, d_gyro)
    d_theta = torch.where(init, 0.0, d_theta)
    self._angle_prev = angle
    self._angle_prev_ok = proj_ok
    self.theta += d_theta
    self.theta_max = torch.maximum(self.theta_max, self.theta)
    self.theta_min = torch.minimum(self.theta_min, self.theta)

    contact = self.foot_contact()
    n_c = contact.sum(-1)
    self.air_cnt = torch.where(n_c == 0, self.air_cnt + 1, 0)
    h = self.base_height()
    # T3 needs 2 airborne steps, by which time the base has already risen
    # ~2 v* dt; remember the height at the first airborne step instead.
    self._h_liftoff = torch.where(self.air_cnt == 1, h, self._h_liftoff)
    vz = data.root_link_lin_vel_w[:, 2]

    # Command input before the trigger (EV-7); ignored once latched (EV-2).
    change = ~self.triggered & (t >= self.cmd_change_time)
    self.cmd_idx = torch.where(change, self.cmd_change_idx, self.cmd_idx)
    self.cmd_change_time = torch.where(change, math.inf, self.cmd_change_time)

    # Trigger with the EV-6 stability interlock.
    # EV-6: first-order low-pass filtered signals (reset to the current value).
    a = min(1.0, dt / C.STABLE_FILTER_TAU_S)
    ang = data.root_link_ang_vel_b.norm(dim=-1)
    jv = data.joint_vel.abs().mean(dim=-1)
    self.filt_ang_vel = torch.where(init, ang, self.filt_ang_vel + a * (ang - self.filt_ang_vel))
    self.filt_joint_vel = torch.where(init, jv, self.filt_joint_vel + a * (jv - self.filt_joint_vel))
    stable = (self.filt_ang_vel < C.STABLE_MAX_ANG_VEL) & (
      self.filt_joint_vel < C.STABLE_MAX_JOINT_VEL
    )
    self.stable_time = torch.where(stable, self.stable_time + dt, 0.0)
    accept = (
      (self.phase == C.STAND)
      & ~self.triggered
      & (t >= self.trigger_input_time)
      & ((self.stable_time >= C.STABLE_HOLD_S) | (not self.interlock))
    )
    self.triggered |= accept
    self.trigger_time = torch.where(accept, t, self.trigger_time)

    # Forward-only transitions, at most one per step.
    p = self.phase
    nxt = p.clone()
    t1 = (p == C.STAND) & accept
    t2 = (p == C.CROUCH) & (h < self.crouch_height)
    t3 = (p == C.LAUNCH) & (self.air_cnt >= C.TAKEOFF_AIR_STEPS)
    t4 = (p == C.AIR) & (self.air_steps >= C.TOUCHDOWN_MIN_AIR_STEPS) & (n_c >= 1)
    upright = -g[:, 2] >= C.UPRIGHT_MIN_COS
    settled = (data.root_link_lin_vel_w.norm(dim=-1) < C.SETTLED_MAX_LIN_VEL) & (
      data.root_link_ang_vel_b.norm(dim=-1) < C.SETTLED_MAX_ANG_VEL
    )
    landed_ok = (
      (n_c == contact.shape[-1])
      & (self.theta >= 2 * math.pi - lvl.success_theta_margin)
      & upright
      & settled
    )
    self.land_hold = torch.where((p == C.LAND) & landed_ok, self.land_hold + dt, 0.0)
    t5 = (p == C.LAND) & (self.land_hold >= lvl.success_hold_s)
    # The first step after a reset only initializes the angle reference.
    t1, t2, t3, t4, t5 = (m & ~init for m in (t1, t2, t3, t4, t5))
    nxt[t1] = C.CROUCH
    nxt[t2] = C.LAUNCH
    nxt[t3] = C.AIR
    nxt[t4] = C.LAND
    nxt[t5] = C.DONE

    # Latches at the transitions.
    self.h_to = torch.where(t3, self._h_liftoff, self.h_to)
    self.H_max = torch.where(t3, h - self._h_liftoff, self.H_max)
    self.theta_td = torch.where(t4, self.theta, self.theta_td)
    self.f_flip = torch.where(
      t4, (self.theta / (2 * math.pi - C.F_FLIP_MARGIN)).clamp(0.0, 1.0), self.f_flip
    )
    self.success |= t5
    self.success_now = t5
    self.success_time = torch.where(t5, t, self.success_time)

    # Air bookkeeping: steps since entering Air, rise above takeoff, apex.
    in_air = nxt == C.AIR
    self.air_steps = torch.where(in_air, torch.where(t3, 0, self.air_steps + 1), 0)
    rise = h - self.h_to
    self.H_max = torch.where(in_air, torch.maximum(self.H_max, rise), self.H_max)
    self.apex_now = (
      (p == C.AIR) & ~t4 & ~self.apex_done & (self._vz_prev > 0.0) & (vz <= 0.0)
    )
    self.apex_done |= self.apex_now
    self._vz_prev = vz

    moved = nxt != p
    self.entered[moved, nxt[moved]] = True
    self.phase_time = torch.where(moved, 0.0, self.phase_time + dt)
    self.phase = nxt
    self.max_phase = torch.maximum(self.max_phase, self.phase)
    self._needs_init = torch.zeros_like(init)

    # EV-14 trackers.
    root_xy = data.root_link_pos_w[:, :2]
    self.start_xy = torch.where(init[:, None], root_xy, self.start_xy)
    self.start_yaw = torch.where(init, data.heading_w, self.start_yaw)
    self.last_xy = root_xy.clone()
    self.last_yaw = data.heading_w.clone()
    self.phase_dur[torch.arange(self.num_envs, device=self.device), self.phase] += torch.where(
      init, 0.0, dt
    )
    trig = self.triggered
    tau = data.actuator_force.abs().amax(dim=-1)
    qd = data.joint_vel.abs().amax(dim=-1)
    self.torque_peak = torch.where(trig, torch.maximum(self.torque_peak, tau), self.torque_peak)
    self.joint_vel_peak = torch.where(trig, torch.maximum(self.joint_vel_peak, qd), self.joint_vel_peak)
    acc = self._env.scene[self.cfg.imu_acc_sensor_name].data.norm(dim=-1)
    self.land_acc_peak = torch.where(
      self.phase == C.LAND, torch.maximum(self.land_acc_peak, acc), self.land_acc_peak
    )

    self.metrics["triggered"] = self.triggered.float()
    self.metrics["success"] = self.success.float()
    self.metrics["max_phase"] = self.max_phase.float()
    self.metrics["theta_td"] = self.theta_td
    self.metrics["H_max"] = self.H_max

  ##
  # CommandTerm hooks.
  ##

  def _episode_time(self) -> torch.Tensor:
    return self._env.episode_length_buf.float() * self._env.step_dt

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Called on reset (resampling_time_range is effectively infinite)."""
    n, dev = len(env_ids), self.device
    allowed = torch.tensor(self.allowed_commands, device=dev)

    def draw_cmd() -> torch.Tensor:
      return allowed[torch.randint(len(allowed), (n,), device=dev)]

    def jitter() -> torch.Tensor:
      steps = torch.randint(self.cfg.input_jitter_steps + 1, (n,), device=dev)
      return steps.float() * self._env.step_dt

    if self.cfg.fixed_command is not None:
      self.cmd_idx[env_ids] = self.cfg.fixed_command
    else:
      self.cmd_idx[env_ids] = draw_cmd()

    # EV-3/4/8: trigger input time (inf for control episodes).
    delay = torch.empty(n, device=dev).uniform_(*self.cfg.trigger_delay_range_s)
    no_trigger = torch.rand(n, device=dev) < self.cfg.no_trigger_prob
    self.trigger_input_time[env_ids] = torch.where(no_trigger, math.inf, delay + jitter())

    # EV-7: change the command once before the trigger.
    change = torch.rand(n, device=dev) < self.cfg.command_change_prob
    if self.cfg.fixed_command is not None:
      change[:] = False
    change_at = torch.rand(n, device=dev) * delay + jitter()
    self.cmd_change_time[env_ids] = torch.where(change, change_at, math.inf)
    self.cmd_change_idx[env_ids] = draw_cmd()

    for buf in (
      self.stable_time, self.trigger_time, self.phase_time, self.theta, self.theta_max,
      self.theta_min, self.land_hold, self.h_to, self.H_max, self.theta_td, self.f_flip,
      self.success_time, self._vz_prev, self._h_liftoff, self.torque_peak,
      self.joint_vel_peak, self.land_acc_peak,
    ):
      buf[env_ids] = 0.0
    for buf in (self.phase, self.max_phase, self.air_cnt, self.air_steps):
      buf[env_ids] = 0
    for buf in (self.triggered, self.success, self.apex_now, self.apex_done, self.success_now):
      buf[env_ids] = False
    self.entered[env_ids] = False
    self.phase_dur[env_ids] = 0.0
    self._needs_init[env_ids] = True

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    """Log per-episode rates over the finished episodes (EV-13), then reset.

    Rates and the EV-14 quantities use only episodes whose trigger was
    accepted, so the control episodes (EV-4) do not dilute them. Keys appear
    as Metrics/flip/<key>.
    """
    assert isinstance(env_ids, torch.Tensor)
    extras: dict[str, float] = {}
    trig = self.triggered[env_ids]
    if trig.any():
      reached = self.max_phase[env_ids][trig]
      extras["rate/air"] = (reached >= C.AIR).float().mean().item()
      extras["rate/land"] = (reached >= C.LAND).float().mean().item()
      extras["rate/success"] = self.success[env_ids][trig].float().mean().item()
      cmd = self.cmd_idx[env_ids][trig]
      succ = self.success[env_ids][trig]
      for i, name in enumerate(C.COMMAND_NAMES):
        mask = cmd == i
        if mask.any():
          extras[f"rate/success_{name}"] = succ[mask].float().mean().item()
      # EV-14.
      ids = env_ids[trig]
      reached = self.max_phase[ids]
      air, land = reached >= C.AIR, reached >= C.LAND
      if air.any():
        err = (self.H_max[ids] - self._dh[self.cmd_idx[ids]]).abs()
        extras["ev14/height_err"] = err[air].mean().item()
      if land.any():
        extras["ev14/theta_td_err"] = (self.theta_td[ids] - 2 * math.pi).abs()[land].mean().item()
        extras["ev14/land_acc_peak"] = self.land_acc_peak[ids][land].mean().item()
      extras["ev14/torque_peak"] = self.torque_peak[ids].mean().item()
      extras["ev14/joint_vel_peak"] = self.joint_vel_peak[ids].mean().item()
      drift = (self.last_xy[ids] - self.start_xy[ids]).norm(dim=-1)
      extras["ev14/xy_drift"] = drift.mean().item()
      dyaw = wrap_to_pi(self.last_yaw[ids] - self.start_yaw[ids]).abs()
      extras["ev14/yaw_change"] = dyaw.mean().item()
      for i, name in enumerate(C.PHASE_NAMES):
        extras[f"phase_time/{name}"] = self.phase_dur[ids, i].mean().item()
    extras.update(super().reset(env_ids))
    return extras

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      self.update_state()

  def _update_metrics(self) -> None:
    pass  # Updated inside update_state().


@dataclass(kw_only=True)
class FlipCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  contact_sensor_name: str = "feet_ground_contact"
  imu_acc_sensor_name: str = "robot/imu_lin_acc"
  robot_spec: RobotSpec = field(default_factory=lambda: GO2_SPEC)
  # Never resample inside an episode: one command and one flip per episode.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  trigger_delay_range_s: tuple[float, float] = C.TRIGGER_DELAY_RANGE_S
  no_trigger_prob: float = C.NO_TRIGGER_PROB
  command_change_prob: float = C.COMMAND_CHANGE_PROB
  input_jitter_steps: int = C.INPUT_JITTER_STEPS
  # EV-6 stability interlock. On for play / evaluation / the real robot; the
  # training curriculum switches it per stage.
  interlock: bool = True
  # Debug / curriculum overrides (None = automatic).
  fixed_command: int | None = None
  fixed_stage: int | None = None
  fixed_level: int | None = None

  def build(self, env: ManagerBasedRlEnv) -> FlipCommand:
    return FlipCommand(self, env)
