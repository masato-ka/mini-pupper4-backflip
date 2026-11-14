# Third-party notices

## unitree_rl_mjlab

- Source: https://github.com/unitreerobotics/unitree_rl_mjlab (commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`)
- License: Apache License 2.0 (`src/mjlab_backflip/robots/unitree_go2/LICENSE.unitree_rl_mjlab`)
- Vendored files:
  - `src/assets/robots/unitree_go2/xmls/` → `src/mjlab_backflip/robots/unitree_go2/xmls/` (unchanged)
  - `src/assets/robots/unitree_go2/go2_constants.py` → `src/mjlab_backflip/robots/unitree_go2/go2_constants.py` (ported to mjlab 1.6.0; changes listed in the module docstring)
  - `src/tasks/velocity/config/go2/{env_cfgs,rl_cfg}.py` → `src/mjlab_backflip/tasks/velocity/` (flat variant only, rebased on mjlab 1.6.0)
