"""Stage / level curriculum S1(L0) -> S2(L0) -> S3(L1) -> S4(L2).

Promotion uses achievement rates, not normalized reward (decided
2026-11-14, see docs/implementation_plan.md):

  S1 -> S2: rotation rate   — touchdown with theta_td >= 2 pi - margin(level)
  S2 -> S3: success rate    — Mid only
  S3 -> S4: min over commands of the per-command success rate

Rates are EMAs over finished episodes whose trigger was accepted (control
episodes are excluded). Each policy iteration (num_steps_per_env env steps),
a rate >= rho extends the "above" streak and a rate < 0.5 rho extends the
"below" streak. Promote after N_up iterations above, demote after N_down
iterations below; either needs min_iters in the current stage.

Per stage the term also sets:
  - the command set and termination level (FlipCommand.set_stage),
  - the Land reward weights (0 in S1),
  - the success_bonus weight (0 in S1, linear ramp 0 -> 20 in S2),
  - the DR ranges (narrow in S1-S2, full from S3),
  - the EV-6 stability interlock (off before ``interlock_from_stage``, S4 by
    default). With the interlock on from the start, the policy never got
    the trigger accepted: exploration noise kept the body moving, and
    staying untriggered (+alive, no failure risk) paid more than flipping.

The curriculum runs at the start of every reset, before the command and
event resets, so it sees the finished episodes' flip state and its DR
changes apply to the same reset.

Note: the stage is not saved in checkpoints. When resuming, pass
``--env.curriculum.flip.params.start-stage <i>`` (0 = S1). To pin a stage for
debugging, ``--env.curriculum.flip.params.fixed-stage <i>``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab_backflip.tasks.backflip import constants as C
from mjlab_backflip.tasks.backflip.mdp.flip_command import FlipCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

LAND_REWARDS = (
  "upright", "land_height", "feet_all_contact", "land_damp", "land_effort", "land_action_rate",
)
SUCCESS_REWARD = "success_bonus"

# Event term -> (params key, DR table key, how to build the value).
_DR_EVENTS = {
  "encoder_bias": ("bias_range", "encoder_bias"),
  "foot_friction": ("ranges", "foot_friction"),
  "added_mass": ("ranges", "added_mass"),
  "base_com": ("ranges", "com_offset"),
}


class flip_curriculum:
  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self._env = env
    self._flip = env.command_manager.get_term(p.get("command_name", "flip"))
    assert isinstance(self._flip, FlipCommand)
    rm = env.reward_manager
    self._land_weights = {
      n: rm.get_term_cfg(n).weight for n in LAND_REWARDS if n in rm.active_terms
    }
    self._success_weight = rm.get_term_cfg(SUCCESS_REWARD).weight
    self._ema_alpha = 1.0 / p.get("ema_episodes", 2000)
    self._interlock_from = p.get("interlock_from_stage", 3)
    self._rates = torch.zeros(4, device=env.device)  # rotation, success, low/mid/high min
    self._cmd_success = torch.zeros(3, device=env.device)
    self._stage = -1
    self._stage_start_iter = 0
    self._last_iter = -1
    self._above = 0
    self._below = 0
    start = p.get("fixed_stage", -1)
    if start < 0:
      start = p.get("start_stage", 0)
    self._set_stage(int(start), self._iteration(p))

  ##
  # Helpers.
  ##

  def _iteration(self, p) -> int:
    return self._env.common_step_counter // p.get("num_steps_per_env", 24)

  def _ema(self, current: torch.Tensor, value: torch.Tensor, count: int) -> torch.Tensor:
    # Count-weighted EMA: k episodes move the estimate like k single updates.
    k = 1.0 - (1.0 - self._ema_alpha) ** count
    return current + k * (value - current)

  def _metric(self) -> torch.Tensor:
    if self._stage == 0:
      return self._rates[0]
    if self._stage == 1:
      return self._cmd_success[C.MID]
    return self._cmd_success.min()

  def _set_stage(self, stage: int, iteration: int) -> None:
    stage = max(0, min(stage, len(C.STAGES) - 1))
    if stage == self._stage:
      return
    self._stage = stage
    self._stage_start_iter = iteration
    self._above = self._below = 0
    self._rates.zero_()
    self._cmd_success.zero_()
    self._flip.set_stage(stage)
    self._flip.interlock = stage >= self._interlock_from
    # Land terms off in S1.
    rm = self._env.reward_manager
    for name, w in self._land_weights.items():
      rm.get_term_cfg(name).weight = 0.0 if stage == 0 else w
    # DR: narrow in S1-S2, full from S3.
    table = C.DR_FULL if stage >= 2 else C.DR_NARROW
    em = self._env.event_manager
    for event, (key, dr_key) in _DR_EVENTS.items():
      if event not in em.active_terms.get("reset", []):
        continue
      rng = table[dr_key]
      value = {i: rng for i in range(3)} if event == "base_com" else rng
      em.get_term_cfg(event).params[key] = value
    if "pd_gains" in em.active_terms.get("reset", []):
      params = em.get_term_cfg("pd_gains").params
      params["kp_range"], params["kd_range"] = table["kp_scale"], table["kd_scale"]

  def _success_bonus_weight(self, iteration: int, ramp_iters: int) -> float:
    if self._stage == 0:
      return 0.0
    if self._stage == 1:
      frac = min(1.0, (iteration - self._stage_start_iter) / max(1, ramp_iters))
      return frac * self._success_weight
    return self._success_weight

  ##
  # Curriculum hook.
  ##

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    command_name: str = "flip",
    thresholds: tuple[float, float, float] = (0.4, 0.4, 0.7),
    promote_iters: int = 100,
    demote_iters: int = 200,
    min_iters: int = 500,
    success_ramp_iters: int = 1000,
    num_steps_per_env: int = 24,
    ema_episodes: int = 2000,
    start_stage: int = 0,
    fixed_stage: int = -1,
    interlock_from_stage: int = 3,
  ) -> dict[str, float]:
    flip = self._flip
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

    # Update the rate EMAs from the finished, triggered episodes. At the
    # first reset (episode length 0) nothing has finished yet.
    done = env_ids[(env.episode_length_buf[env_ids] > 0) & flip.triggered[env_ids]]
    if len(done) > 0:
      margin = flip.level_cfg.success_theta_margin
      rotated = (flip.max_phase[done] >= C.LAND) & (flip.theta_td[done] >= 2 * math.pi - margin)
      self._rates[0] = self._ema(self._rates[0], rotated.float().mean(), len(done))
      self._rates[1] = self._ema(self._rates[1], flip.success[done].float().mean(), len(done))
      cmd = flip.cmd_idx[done]
      for i in range(3):
        mask = cmd == i
        if mask.any():
          rate = flip.success[done][mask].float().mean()
          self._cmd_success[i] = self._ema(self._cmd_success[i], rate, int(mask.sum()))

    # Once per policy iteration: streaks, promotion, demotion.
    it = self._iteration({"num_steps_per_env": num_steps_per_env})
    if it != self._last_iter:
      self._last_iter = it
      if fixed_stage >= 0:
        self._set_stage(fixed_stage, it)
      elif self._stage < len(C.STAGES):
        rho = thresholds[min(self._stage, len(thresholds) - 1)]
        metric = float(self._metric())
        self._above = self._above + 1 if metric >= rho else 0
        self._below = self._below + 1 if metric < 0.5 * rho else 0
        stayed = it - self._stage_start_iter >= min_iters
        last = len(C.STAGES) - 1
        if stayed and self._stage < last and self._above >= promote_iters:
          self._set_stage(self._stage + 1, it)
        elif stayed and self._stage > 0 and self._below >= demote_iters:
          self._set_stage(self._stage - 1, it)

    w = self._success_bonus_weight(it, success_ramp_iters)
    env.reward_manager.get_term_cfg(SUCCESS_REWARD).weight = w

    return {
      "stage": float(self._stage + 1),  # 1..4 = S1..S4
      "level": float(flip.level),
      "metric": float(self._metric()),
      "rotation_rate": float(self._rates[0]),
      "success_rate": float(self._rates[1]),
      "success_low": float(self._cmd_success[C.LOW]),
      "success_mid": float(self._cmd_success[C.MID]),
      "success_high": float(self._cmd_success[C.HIGH]),
      "above_streak": float(self._above),
      "below_streak": float(self._below),
      "success_bonus_weight": w,
      "interlock": float(flip.interlock),
    }
