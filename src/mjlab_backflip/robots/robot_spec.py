"""Robot-dependent values used by the backflip task.

Every number here is provisional (Go2 stand-in for mini-pupper 4). When the
real robot spec is available, replace this module and the asset package only.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
  # Names in the MJCF.
  base_body: str
  imu_site: str
  foot_names: tuple[str, ...]  # Site names, also the foot geom prefix.
  base_geom_names: tuple[str, ...]  # Torso collision geoms (TM-2).

  # Base height with the default pose and the feet just touching the ground
  # [m]. Used as the reset height and as h_nom in the rewards and phase
  # thresholds. Measured by scripts/measure_stand_height.py.
  h_nom: float

  # IMU site position relative to the base origin [m] (top center of torso).
  imu_pos: tuple[float, float, float]

  # Sign that maps the body-frame pitch rate omega_y to the backflip
  # direction (nose up first): omega_back = back_pitch_sign * omega_y.
  # Verified by scripts/check_signs.py and tests/test_signs.py.
  back_pitch_sign: float

  def back_pitch_angle(self, g_x, g_z):
    """Pitch angle of the body-frame gravity, positive in the backflip
    direction. Accumulate its wrapped increments to get theta.

    The spec writes phi = atan2(g_x, -g_z); on Go2 that angle decreases when
    the nose goes up, so the backflip-positive angle is sign * phi.
    """
    return self.back_pitch_sign * _atan2(g_x, -g_z)


def _atan2(y, x):
  # Works for floats and torch tensors.
  if isinstance(y, float):
    return math.atan2(y, x)
  import torch

  return torch.atan2(y, x)


GO2_SPEC = RobotSpec(
  base_body="base_link",
  imu_site="imu",
  foot_names=("FR", "FL", "RR", "RL"),
  base_geom_names=("base1_collision", "base2_collision", "base3_collision"),
  # 2026-11-14 measurement: kinematic 0.295 m. With zero action and the
  # unitree PD gains the base sags to 0.224 m (11 deg nose-up, rear hips
  # splayed 0.29 rad), so holding h_nom needs active control.
  h_nom=0.295,
  imu_pos=(0.0, 0.0, 0.05),
  # Nose up = negative euler pitch = negative omega_y on Go2.
  back_pitch_sign=-1.0,
)
