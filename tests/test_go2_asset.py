"""CPU checks for the vendored Go2 asset and the sanity task registration."""

import numpy as np

import mjlab  # noqa: F401  (loads the mjlab.tasks entry points)
from mjlab.entity import Entity
from mjlab.tasks.registry import list_tasks, load_env_cfg

from mjlab_backflip.robots import GO2_SPEC, get_go2_robot_cfg


def test_go2_compiles_with_moved_imu():
  robot = Entity(get_go2_robot_cfg())
  model = robot.spec.compile()
  assert model.nu == 12
  site_id = model.site(GO2_SPEC.imu_site).id
  np.testing.assert_allclose(model.site_pos[site_id], GO2_SPEC.imu_pos)


def test_go2_velocity_task_registered():
  assert "Mjlab-Velocity-Flat-Unitree-Go2" in list_tasks()
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-Go2")
  assert "robot" in cfg.scene.entities
