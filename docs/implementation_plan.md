# Go2 バックフリップ強化学習（mjlab）実装方針と計画

## Context

mini-pupper 4 のバックフリップ方策を mjlab で学習させたい。実機と設計モデルはまだないので、先に Unitree Go2 で仕様書の内容（5相の状態機械、非対称 Actor-Critic、報酬、終了条件、カリキュラム）が成り立つかを確かめる。仕様書は `四足バックフリップ強化学習 実装仕様書(mjlab).md` で、実装はこれに従う。

学習は mjlab の `train` コマンドで回し、同じ内容を Google Colab の Notebook からも実行できるようにする。後で mini-pupper 4 へ移るときは、ロボット資産と機体依存の定数を差し替えるだけで済む構成にする。

**確定事項**（ユーザ回答）
- mjlab は **1.6.0** に固定する。
- **unitree_rl_mjlab**（Apache-2.0、mjlab 1.2.0 向け）から Go2 の MJCF、メッシュ、`go2_constants.py`、velocity Go2 cfg の該当部分をベンダリングする。これらを 1.6.0 の API に移植し、自前パッケージから entry point で登録する。

**調査で分かったこと**
- mjlab 本体（1.3.0、1.6.0 main とも）には Go2 がない。unitree_rl_mjlab の `src/assets/robots/unitree_go2/` が Go2 の唯一の出どころになる。
  - `go2.xml` は胴体を `base_link` とする。IMU サイトは `imu` で、その上に `gyro imu_ang_vel` と `accelerometer imu_lin_acc` がある。
  - 足のジオムは `{FR,FL,RR,RL}_foot_collision`、足先サイトは `FR/FL/RR/RL`。
- unitree_rl_mjlab の値は Kp 20/20/40、Kd 1/1/2、effort 23.5/23.5/45、初期 z 0.32、`soft_joint_pos_limit_factor=0.9`。PD ゲインはこの値を採用する。
- mjlab の API で使うもの
  - CommandTerm の hook: `_resample_command(env_ids)`、`_update_command()`、`_update_metrics()`。1.6.0 では `_update_command(self, env_ids)` に変わっている可能性があり、M0 で確認する。
  - 観測項目ごとの設定: `ObservationTermCfg` の `history_length`、`delay_min_lag/max_lag`、`noise`、`clip`。
  - アクチュエータの行動遅れ: `ActuatorCfg.delay_*_lag`。物理ステップ単位で指定する。
  - 終了条件: `TerminationTermCfg(time_out=True)`。
  - ドメインランダム化: `mdp.dr.*`（`geom_friction`、`body_mass`、`body_com_offset`、`pd_gains`、`encoder_bias`）。
  - ログ: `env.extras["log"]["Metrics/..."]`。
- 注意点（microduck_rl の経験から）
  - velocity 基底の `fell_over`（bad_orientation）はフリップ中に必ず発火するので外す。
  - 報酬は `scale_rewards_by_dt=True` のため、疎ボーナスも dt 倍される。
  - 実行時に cfg を書き換えるときは `get_term_cfg` を経由する。managers は cfg を deepcopy している。
  - mjlab 1.x は `scipy` の依存宣言が漏れているので、自前で追加する。
  - Colab では `MUJOCO_GL=egl` が必要。

## 構成（新規リポジトリ `mini-pupper4-backflip/`）

```
pyproject.toml            # uv_build, src layout, python>=3.10,<3.14
                          # deps: mjlab==1.6.0, scipy, (torch は mjlab 経由)
                          # [project.entry-points."mjlab.tasks"] mjlab_backflip = "mjlab_backflip.tasks"
THIRD_PARTY_NOTICES.md    # unitree_rl_mjlab (Apache-2.0) の出典とライセンス
src/mjlab_backflip/
  robots/unitree_go2/     # ベンダリング: xmls/go2.xml, assets/*.obj, go2_constants.py(1.6移植)
  robots/robot_spec.py    # 機体依存値の集約 (h_nom, Kp/Kd, effort/vel 上限, IMU オフセット, q_tuck, 足名)
                          #   → mini-pupper 4 移行時はここと robots/ だけ差し替える
  tasks/__init__.py       # register_mjlab_task(...)
  tasks/backflip/
    constants.py          # 仕様書の数値表: コマンド表(Δh*, v*, t_f*, ω*, ρ_c), 重み, L0–L2 しきい値, S1–S4
    backflip_env_cfg.py   # make_backflip_env_cfg(play=False) : Go2 velocity cfg を土台に組み替え
    rl_cfg.py             # PPO 設定 (Go2 velocity rl_cfg を流用, experiment_name="go2_backflip")
    mdp/
      flip_command.py     # FlipCommand(CommandTerm): 状態機械・コマンド・トリガー (中核)
      observations.py     # Actor/Critic 観測関数, IMU 推定ノイズモデル
      rewards.py          # phase_gated と相専用項・共通項
      terminations.py     # TM-2〜TM-9
      curriculum.py       # ステージ S1–S4 / 終了レベル L0–L2 の進級・降格
      events.py           # リセット初期化 (実測立ち高さ), DR パラメータ記録
      metrics.py          # 相別報酬・滞在時間, コマンド別成功率, EV-14
scripts/
  measure_stand_height.py # 実際の立ち高さ h_nom を測る (初期 z の不一致対策)
  check_state_machine.py  # 学習なしの開ループ入力で相遷移と符号を検証
  check_signs.py          # φ=atan2(g_x,−g_z) と ω_back の符号を確定
tests/test_backflip_cfg.py# CPU で cfg 構築, 項の存在・重みの符号, 観測次元=223
notebooks/backflip_go2_colab.ipynb
```

## 主要コンポーネントの実装方針

### 1. ロボット資産（Go2 を 1.6.0 へ移植）
- unitree の `go2_constants.py` をコピーする。import 先を `mjlab_backflip` に変え、`BuiltinPositionActuatorCfg` などを 1.6.0 の API に合わせる。
- Kp と Kd は **unitree サンプルの値（hip/thigh Kp 20・Kd 1、calf Kp 40・Kd 2）をそのまま使う**（2026-11-14 決定）。`robot_spec.py` の定数として持ち、実機へ差し替えるときに変更する。仕様書の Kp 70 / Kd 3 は採用しない。
- 行動遅れ 0.02 s はアクチュエータの `delay_min_lag=delay_max_lag=4`（timestep 0.005 × 4）で入れる。ランダム化は S3 以降に回す。
- IMU は、`spec_fn` 内で MjSpec を編集し、`imu` サイトを胴体上面の中央（`pos=(0,0,+0.05)`、定数）へ移す。センサ名は `robot/imu_ang_vel` と `robot/imu_lin_acc` を使う。
- `init_state.pos.z` には `measure_stand_height.py` の実測値を入れる。

### 2. FlipCommand（状態機械とコマンドの中核）
- 環境ごとに次の状態を持つ。
  - コマンド: `height_idx`（0/1/2）、`cmd_scalar`（0/0.5/1）。
  - トリガー: `trigger_time`、`trigger`（ラッチ）、`t_since_trigger`、`stable_timer`（EV-6）。
  - 相と回転: `phase`（0 Stand、1 Crouch、2 Launch、3 Air、4 Land、5 Done）、`phase_time`、`theta`、`phi_prev`。
  - 遷移と着地の記録: `air_cnt`、`air_time`、`land_hold`、`h_to`、`H_max`、`apex_paid`、`f_flip`、`success`、`success_time`。
- `_resample_command(env_ids)` はリセット時に呼ばれ、次を行う。`resampling_time_range` を大きくして、エピソード中には再抽選しない。
  - EV-1: コマンドを抽選する。ステージで Mid のみか3種かを切り替える。
  - EV-3: トリガー遅れを U[0.5, 1.5] で抽選する。
  - EV-4: 15% のエピソードではトリガーを出さない。
  - EV-7: 20% の確率で、トリガー前に変更する時刻と変更先の値を決めておく。
  - 相の状態はすべて初期化する。
- 状態は毎ステップ `update_state()` で更新する。**同じステップで1回だけ**実行されるよう、`env.common_step_counter` で守る。
  - 遷移 T1〜T5 を判定する。前進のみとする。T5 の条件はレベルに応じたしきい値で評価する。
  - θ の積分、h_to のラッチ、f_flip の接地時ラッチ、成功のラッチ（EV-9、EV-10）、EV-8 の入力ジッタも、ここで扱う。
  - 呼び出し元は終了条件の関数の先頭と `_update_command`。
  - こうする理由: mjlab の step は「終了判定 → 報酬 → … → command.compute」の順に進む。compute だけで状態を更新すると、報酬と終了判定が1ステップ古い相を見てしまう。
- `command` プロパティは、Actor に渡すスカラー群 `[cmd_scalar, trigger, t_since_trigger/2]` を返す。報酬や観測の関数は `env.command_manager.get_term("flip")` の属性を直接読む。
- 接地は `feet_ground_contact`（`ContactSensorCfg`、`found`）から取る。胴体の接地を見るために、`base_contact` センサも追加する。

### 3. 観測（`actor` 223 次元、`critic` 約100 次元）
- **Actor の履歴項**（`history_length=4`、それぞれ観測遅れ 1〜2 ステップ）
  - gyro: `builtin_sensor("robot/imu_ang_vel")`。バイアスとスケール誤差を入れ、±2000 dps で clip する。
  - acc: `builtin_sensor("robot/imu_lin_acc")`。±16 g で clip する。
  - `joint_pos_rel`、`joint_vel_rel`、`last_action`。
  - トルク推定: `asset.data.actuator_force` にノイズを加えたもの。
  - 1ステップ 54 次元、履歴込みで 216 次元。
- **Actor の非履歴項**
  - g_est と θ_est/2π: 真値にノイズとバイアスのランダムウォークを加える。クラス型の観測項にして、`reset` でバイアスを再抽選する。
  - cmd_scalar、trigger、t_since_trigger。
  - 合計 7 次元。
- ノイズの基準値は Genesis の値（ang_vel 0.1、gravity 0.02、dof_pos 0.01、dof_vel 0.5）。`NoiseModelWithAdditiveBiasCfg` と `Unoise` を使う。
- **Critic**: 仕様書の列挙のとおり。
  - Actor の観測（ノイズなし）。
  - 真の状態: 速度、高さ、重力方向、θ。
  - 相の情報: 相の one-hot、相の経過時間、τ_air。
  - 接地と力: 接地フラグ、接地力、足先高さ、トルク。
  - フリップ関連: h_to、H、v\*、t_f\*、f_flip、トリガー。
  - DR パラメータ: 摩擦、付加質量、重心のずれ、Kp/Kd 倍率、エンコーダーバイアスの代表値（約8個）。`events.py` でリセット時に env 側のバッファへ記録したものを読む。

### 4. 行動
- `JointPositionActionCfg(scale=0.5, use_default_offset=True)`。50 Hz、decimation 4、timestep 0.005。
- エピソード長は 5.0 s。

### 5. 報酬
- `phase_gated(env, phase_id, term_fn, **kw)` を汎用のラッパー関数にする。`phase_id` には int か範囲（Launch〜Land）を渡せるようにする。
- 項は仕様書の表どおりに一対一で関数化する。
  - 共通項 6 個。
  - Stand 4 個、Crouch 4 個、Launch 4 個、Air 4 個（tuck と land_prep は任意で、既定 OFF）、Land 7 個。
  - Land の正の項には f_flip をかける。
- 疎の項（apex_bonus、success_bonus）は、関数側で発火した1ステップだけ `1/env.step_dt` を返し、dt スケールを打ち消す。
- 符号の規約: 関数は正の量を返し、重みの符号で減点か加点かを表す。テストで固定する。
- terminal_penalty は `mdp.is_terminated` を重み −5 で使う。time_out は除外されることを確認する。
- 重みはすべて `constants.py` に置き、tyro の CLI で上書きできるようにする。

### 6. 終了条件
- velocity 由来の `fell_over`、`illegal_contact`、`out_of_terrain_bounds` は削除する。
- 失敗: TM-2（胴体接触）、TM-3（高さ、リセット後20ステップの猶予）、TM-4、TM-5、TM-6、TM-7（相の滞在上限 × レベル倍率）、TM-8（非有限、または 20 m/s 超）。
- 打ち切り: TM-1 `time_out`、TM-9（成功後 0.5 s）。どちらも `time_out=True`。
- しきい値は FlipCommand が持つ現在レベル（L0〜L2）の表から引く。

### 7. カリキュラム
- `curriculum.py` の `flip_curriculum` をクラス型の CurriculumTerm にし、ステージ S1〜S4 と終了レベルを1か所で管理する。
  - **進級の指標は報酬ではなく達成率とする（2026-11-14 決定、案C）。** 仕様書の R̂（正規化エピソード報酬）は使わない。
  - ステージと終了レベルは S1(L0) → S2(L0) → S3(L1) → S4(L2) の一本の段階として進める。
  - 達成率は、トリガーを出したエピソードだけを分母にした EMA とする。EV-4 の対照エピソードは除き、立ち続けられた割合を別に記録する。

    | 遷移 | 指標 | しきい値 ρ（初期値） |
    |---|---|---|
    | S1 → S2 | 回転到達率: 接地時の θ が現レベルの成功判定値（L0 は 2π − 1.0）以上 | 0.4 |
    | S2 → S3 | 成功率（EV-9、現レベルの判定。Mid のみ） | 0.4（仕様書の L0→L1） |
    | S3 → S4 | コマンド別成功率の最小値（Low/Mid/High） | 0.7（仕様書の L1→L2） |

  - 進級と降格のルールは仕様書の数字を引き継ぐ。ρ 以上を N=100 反復続けたら進級し、各段階に最低 500 反復は留まる。0.5ρ を 200 反復下回ったら 1 段階戻す。
  - 反復数: `common_step_counter / num_steps_per_env`。
- ステージで切り替えるもの
  - コマンドの集合（FlipCommand のフィールド）。
  - Land 項の重み。`reward_manager.get_term_cfg(name).weight` を書き換える。
  - success_bonus のランプ（0→20 の線形）。
  - DR の幅。S3 から広げる。event の `get_term_cfg().params` を書き換える。
- 記録: `Curriculum/` 配下に、段階、達成率（回転到達率、コマンド別成功率）、Air 到達率、対照エピソードで立ち続けられた割合を出す。
- デバッグ用に、ステージとレベルを固定する CLI フラグを設ける（例: `--env.curriculum.flip.params.fixed_stage 1`）。

### 8. イベントと DR
- リセット
  - 実測の立ち高さ、yaw ∈ [−π, π]、関節ノイズ（小）、速度ゼロ。
  - `push_robot` は削除する。
- DR はリセット時に行う。
  - 関節オフセット ±0.02: `dr.encoder_bias`。
  - 摩擦 0.2〜1.5: `dr.geom_friction`、足のジオム。
  - 付加質量 −1〜3 kg: `dr.body_mass`、operation add。
  - 重心 ±0.01: `dr.body_com_offset`。
  - Kp と Kd ×0.8〜1.2: `dr.pd_gains`。
  - S1 と S2 では範囲を狭める。

### 9. 記録（EV-13〜15、検証の章）
- `Metrics/flip/`
  - 相ごとの累積報酬と滞在時間。
  - コマンド別の成功率。
  - EV-14 の量: |H−Δh\*|、着地時の θ の誤差、トルクと関節速度のピーク、着地加速度のピーク、XY のずれ、yaw の変化。
- いずれも FlipCommand の `metrics` と `extras["log"]` に書く。終了の内訳は mjlab が `Episode_Termination/` に出す。

### 10. タスク登録
- `Mjlab-Backflip-Flat-Unitree-Go2`: 学習用。カリキュラムは自動。
- play_env_cfg: ノイズ OFF、カリキュラムを最終ステージに固定、トリガーを必ず出す。
- runner: `VelocityOnPolicyRunner`。ONNX の書き出しのため。`scene` のキーは `"robot"` を維持する。

## 実装の段取り（マイルストーン）

| # | 内容 | 完了条件 |
|---|---|---|
| M0 | uv プロジェクトの雛形、mjlab 1.6.0 の API 確認（CommandTerm の hook 名、actuator delay、dr の関数名、`is_terminated` の挙動）、Go2 のベンダリングと移植 | `uv run list-envs` に Go2 の velocity 相当のテストタスクが出る。`play --agent zero` で立つ |
| M1 | `measure_stand_height.py`、`check_signs.py`、IMU サイトの移設 | h_nom の実測値と、φ と ω_back の符号を `robot_spec.py` に確定 |
| M2 | FlipCommand（状態機械、トリガー、EV-1〜12）と `check_state_machine.py` | 開ループのしゃがみ→跳躍→（手で回転を与えて）着地で相が 0→5 と単調に進む。逆戻りしない |
| M3 | 観測（223 次元と Critic）、行動、終了条件、イベント、DR | `tests/` が通る（次元と符号）。`--agent random` で各 TM が発火し、内訳がログに出る |
| M4 | 報酬（phase_gated と全項）とメトリクス | `Episode_Reward/*` の符号が規約どおりになる。疎ボーナスが1回だけ入る |
| M5 | カリキュラム（S1〜S4、L0〜L2、ランプ、DR の拡大） | 固定ステージのフラグで各ステージが起動する。短時間学習で進級ログが出る |
| M6 | Colab Notebook | Colab の T4 で、スモーク学習→動画→保存まで通る |
| M7 | 本学習（S1→S4）と重みの調整 | コマンド別の成功率と EV-14 で評価する |

## Colab Notebook（`notebooks/backflip_go2_colab.ipynb`）

mjlab-handson の notebook のセル構成を踏襲する。ただし学習は **CLI の `train` を `!` で呼ぶ**ことで、ローカルと手順を揃える。

1. Open-in-Colab バッジと説明。`!nvidia-smi` で GPU を確認する。
2. インストール: `!pip install -q "git+https://github.com/<user>/mini-pupper4-backflip.git"`（mjlab==1.6.0 が依存で入る）。リポジトリが非公開の間は、Drive の zip から `pip install -e` する方法を代替として載せる。
3. `os.environ["MUJOCO_GL"]="egl"`。wandb へのログインは任意。使わない場合は `--agent.logger tensorboard`。
4. Google Drive をマウントし、`logs/` を Drive へのシンボリックリンクにする。セッションが切れても残すためと、`--agent.resume` で再開するため。
5. スモーク: `!train Mjlab-Backflip-Flat-Unitree-Go2 --env.scene.num-envs 64 --agent.max-iterations 5`。
6. 本学習: num-envs 2048〜4096（T4 で調整）。`--video --video-interval` を付ける。
7. 再生: `!play ... --agent trained --checkpoint-file ... --video`。mediapy で表示する。
8. TensorBoard を表示し、ONNX と checkpoint をダウンロードする。

## 検証（エンドツーエンド）

- **CPU 上（ローカル Mac）**
  - `uv run --with pytest pytest tests/`: cfg の構築、観測 223 次元、報酬の符号、終了項の種別（time_out の有無）。
  - `uv run python scripts/check_state_machine.py`: 相が単調に進むこと、θ の符号、h_to と f_flip のラッチ、トリガーが1回だけ受理されること（EV-6、再トリガーの拒否）。
- **GPU 上（Colab または Linux）**
  - `play Mjlab-Backflip-Flat-Unitree-Go2 --agent zero` と `--agent random`。落ちないこと、各 TM が発火すること。
  - スモーク学習（64 envs × 5 iter）。続けて短時間学習（S1 固定）で、Launch と Air の到達率が上がることを `Metrics/flip/` で確認する。
- **評価**: コマンド別の成功率、EV-14、トルクと関節速度のピーク。Go2 の effort 23.5/45 Nm に対してどれだけ余裕があるか、Low の ω\*=13.9 rad/s を出せるかを見る。

## 未決事項として実装時に確認・記録するもの

- mjlab 1.6.0 での `_update_command` のシグネチャ、`is_terminated` が time_out を含むか、dr の関数名と引数。M0 で確認し、`docs/notes.md` に記録する。
- 達成率のしきい値 ρ は初期値。S1→S2 の 0.4 は仕様書にない値として置いた。S4 の到達条件（最終的な成功率の目標）も仕様書で未決のまま。
