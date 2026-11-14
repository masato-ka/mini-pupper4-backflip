# mini-pupper4-backflip

mjlab で四足ロボットのバックフリップ方策を学習する。mini-pupper 4 の実機仕様が出るまでは、Unitree Go2 のシミュレーションで方策と観測の妥当性を検証する。

- 仕様書: `四足バックフリップ強化学習 実装仕様書(mjlab).md`
- 実装計画: `docs/implementation_plan.md`
- 実装メモ（mjlab 1.6.0 の確認結果、仕様書からの変更点）: `docs/notes.md`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masato-ka/mini-pupper4-backflip/blob/main/notebooks/backflip_go2_colab.ipynb)

## タスク

| Task ID | 内容 |
|---|---|
| `Mjlab-Backflip-Flat-Unitree-Go2` | 1 エピソード 1 回のバックフリップ。ジャンプ高さコマンド Low / Mid / High とトリガー |
| `Mjlab-Velocity-Flat-Unitree-Go2` | 動作確認用の Go2 歩行タスク（unitree_rl_mjlab から移植） |

## セットアップ

```bash
uv sync
uv run list-envs
```

学習には CUDA が必要。CPU（Mac を含む）でも動作確認とデバッグはできる（`--gpu-ids None`）。

## 学習

```bash
# GPU
uv run train Mjlab-Backflip-Flat-Unitree-Go2 --env.scene.num-envs 4096 --agent.logger tensorboard

# CPU での動作確認
uv run train Mjlab-Backflip-Flat-Unitree-Go2 --gpu-ids None --env.scene.num-envs 16 --agent.max-iterations 2 --agent.logger tensorboard
```

主なオプション:

- `--env.curriculum.flip.params.fixed-stage <0..3>`: カリキュラムのステージを固定する（0 = S1）。
- `--env.curriculum.flip.params.start-stage <0..3>`: 開始ステージ。ステージはチェックポイントに保存されないので、`--agent.resume True --agent.load-run <run>` で再開するときに指定する。
- `--env.rewards.<name>.weight <w>`: 報酬の重み。

Google Colab では `notebooks/backflip_go2_colab.ipynb` を使う。

## 確認用スクリプトとテスト

```bash
uv run python scripts/measure_stand_height.py      # 立ち高さの実測
uv run python scripts/check_signs.py               # 回転方向の符号
uv run python scripts/check_state_machine.py --assist-landing   # 状態機械（人工的な宙返り）
uv run python scripts/record_video.py --checkpoint logs/rsl_rl/go2_backflip/<run>/model_<n>.pt
uv run --group dev pytest -q tests
```

## 構成

```
src/mjlab_backflip/
  robots/robot_spec.py        機体依存の値（h_nom、IMU 位置、回転の符号）。mini-pupper 4 へ移るときはここと robots/ を差し替える
  robots/unitree_go2/         Go2 の MJCF、メッシュ、定数（unitree_rl_mjlab から移植）
  tasks/backflip/
    constants.py              仕様書の数値（コマンド表、レベル、ステージ、報酬の重み、DR、センサモデル）
    backflip_env_cfg.py       環境設定
    mdp/flip_command.py       コマンド、トリガー、5 相の状態機械
    mdp/observations.py       Actor（223 次元）と Critic（115 次元）の観測
    mdp/rewards.py            相ごとの報酬
    mdp/terminations.py       終了条件 TM-2〜TM-9
    mdp/curriculum.py         ステージ S1〜S4 と終了レベル L0〜L2
    mdp/events.py             リセット時のイベント
```

## ライセンス

Go2 のアセットと設定の一部は [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)（Apache-2.0）から取り込んでいる。`THIRD_PARTY_NOTICES.md` を参照。
