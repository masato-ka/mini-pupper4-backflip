# 実装メモ（mjlab 1.6.0 の確認結果）

M0 で mjlab 1.6.0 のソースを読んで確認した事項を記録する。

## 依存関係

- mjlab 1.6.0 は mujoco 3.11.0、mujoco-warp 3.11.0、rsl-rl-lib 5.4.2 を引き込む。torch は 2.15.0 に解決された。
- mjviser 0.0.17 は mujoco-core 4.0.1 に依存し、mujoco 3.11 のプラグインを上書きして `import mujoco` を壊す。`pyproject.toml` で `mjviser<0.0.17` に固定した。Colab で `pip install` するときも、このパッケージ経由で入れれば同じ制約がかかる。
- scipy は mjlab が宣言していないので直接依存に入れた。Python は scipy の制約により 3.11 以上。

## API

- `CommandTerm._update_command(self, env_ids)` は `env_ids` 引数が必須。引数なしで定義すると `_check_update_command_signature` が TypeError を出す。毎ステップの更新では `None`、リセット時はリセットした env の id が渡る。
- `CommandTerm.compute(dt, env_ids=None)` は `_update_metrics` → タイマー減算と再抽選 → `_update_command(env_ids)` の順に呼ぶ。自動リセットされた env には dt=0 が渡る。
- `mdp.is_terminated` は `termination_manager.terminated` を返す。time_out=True の項は含まない。terminal_penalty にそのまま使える。
- 1 ステップの順序は次のとおり。
  1. 行動の適用と物理の decimation ループ
  2. `episode_length_buf` と `common_step_counter` を加算
  3. 終了判定 → 報酬 → metrics → step/interval イベント
  4. 自動リセット（curriculum → reset イベント → 各 manager の reset）
  5. `sim.forward()` → `command_manager.compute` → `sim.sense()` → 観測
- 終了判定と報酬は、物理 1 サブステップ分古い派生量（xpos、sensordata など）を見る。`qpos`/`qvel` 由来の `root_link_*` は最新。
- コマンド項は終了判定と報酬のあとで更新される。そのため FlipCommand の状態更新は、終了判定の関数から先に呼び、`common_step_counter` で 1 ステップ 1 回に制限する（計画どおり）。
- 行動遅れ: `ActuatorCfg.delay_min_lag` / `delay_max_lag`（物理ステップ単位）。
- 観測: `ObservationTermCfg.delay_min_lag` / `delay_max_lag`（方策ステップ単位）、`history_length`、`noise`、`clip`。ノイズには `NoiseModelWithAdditiveBiasCfg` がある。
- DR: `dr.pd_gains`、`dr.effort_limits`、`dr.geom_friction`、`dr.encoder_bias`、`dr.body_mass`、`dr.body_com_offset`。
- カリキュラム: `mdp.reward_curriculum` と `mdp.termination_curriculum`（クラス型）がある。
- `CollisionCfg` の構造系フィールド（priority、condim など）を dict で書くときは、該当するすべてのジオムを覆う必要がある。unitree の Go2 設定には `".*": 0` を足した。

## Go2 の挙動（CPU、ゼロ行動、unitree のゲイン）

- 4 env、150 ステップ（3 s）で転倒せず立ち続けた。
- 胴体の高さは約 0.23 m まで沈み、3 s の時点でも少しずつ下がっていた（calf トルクは −4.6 Nm まで増加）。Kp 20/40 では、既定姿勢を目標にしても重力で沈む。h_nom は M1 で実測して決める。初期 z 0.32 は実際の立ち高さより高い。
- 静止時の加速度計は z ≈ +9.65 で、比力として上向きの値が出る。x ≈ +1.8 は胴体が傾いているため。

## CPU での実行（Mac、Apple Silicon）

MuJoCo Warp は NVIDIA Warp の上に作られており、Warp には CPU バックエンド（LLVM でカーネルをコンパイル）がある。CUDA がなくても、device を `cpu` にすれば動く。速度が出ないので、動作確認とデバッグに限って使う。

- Python から: `ManagerBasedRlEnv(cfg=cfg, device="cpu")`
- train コマンド: `uv run train <TASK> --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 2 --agent.logger tensorboard`
  - `--gpu-ids None` にすると `CUDA_VISIBLE_DEVICES=""` になり、device が cpu になる。
  - 8 env × 2 イテレーションで 1 イテレーション約 1.4 s（初回はカーネルのコンパイルに数秒かかる）。
- play コマンド: CUDA がなければ自動で cpu になる。ただしビューアを Mac で開くには mjpython が要る。

## M1 の結果（2026-11-14、CPU）

### 立ち高さ（`scripts/measure_stand_height.py`）

- 既定姿勢で足がちょうど接地する高さ（運動学的な高さ）は 0.295 m。unitree の初期 z 0.32 より 2.5 cm 低い。
- ゼロ行動（unitree の PD ゲインで既定姿勢を保持）では、胴体は 15 s かけて 0.224 m まで沈んで止まる。このとき機首上げ 11°、後脚の hip が外へ 0.29 rad 開き、calf のトルクは最大 9.6 Nm。
- 決定: リセット高さと h_nom は 0.295 m（運動学的な高さ）とした。リセット時の落下はなくなる。0.295 m を保つには方策が能動的に支える必要があるが、calf で τ/Kp ≈ 0.24 rad の目標オフセットに相当し、行動スケール 0.5 の範囲に収まる。
- Crouch の目標 ρ_c·h_nom は Low 0.236、Mid 0.221、High 0.207 m になる。

### 符号（`scripts/check_signs.py`、`tests/test_signs.py`）

- Go2 では、機首上げ（バックフリップの前半）は euler pitch が負、胴体座標の ω_y が負の向き。IMU ジャイロの y 成分も同じ符号。
- 仕様書の φ = atan2(g_x, −g_z) は、機首上げで減る。バックフリップの向きを正にするため、`RobotSpec.back_pitch_sign = −1` を掛ける。
  - ω_back = −ω_y
  - θ の増分 = wrap(Δ(−φ))
- `RobotSpec.back_pitch_angle(g_x, g_z)` がこの符号込みの角度を返す。状態機械はこれを使う。

### 並び順の注意

- `robot.joint_names` は脚ごと（FL_hip、FL_thigh、FL_calf、FR_hip、…）。
- `robot.actuator_names` と `actuator_force`、行動ベクトルは関節種別ごと（FL_hip、FR_hip、RL_hip、RR_hip、FL_thigh、…）。
- トルク観測と actions_symmetry 報酬では、どちらの並びかを明示して扱う。

## M2 の結果（2026-11-14、CPU）

`FlipCommand`（`src/mjlab_backflip/tasks/backflip/mdp/flip_command.py`）を実装した。開ループの確認は `scripts/check_state_machine.py`、テストは `tests/test_flip_command.py`。

### 確認できたこと

- 手で回転を与えた Mid のフリップ（v* = 2.62 m/s、ω* = 11.8 rad/s を Launch 開始時に胴体へ与える）で、相が Stand → Crouch → Launch → Air → Land → Done と1段ずつ進み、戻らない。
  - 着地補助（接地の瞬間に胴体を直立に戻す）ありでは毎回成功する。
  - 補助なしでも θ は 362° に達するが、制御なしの着地では静止せず、成功判定には届かない。
- EV-4: トリガーなしの対照エピソードは Stand のまま。
- EV-6: 関節を動かし続けている間はトリガーを受理しない。安定してから 0.3 s 後に受理する。
- EV-7、EV-2: トリガー前のコマンド変更は反映される。トリガー後の変更は無視される。観測のスカラーはラッチした値になる。

### 仕様書から変えた点

1. **θ の計算**: 胴体が横に 90° 近く転がると、φ = atan2(g_x, −g_z) は g_x と g_z がともに 0 に近づいて不定になる。確認中に、ピッチ回転なしで θ が 0.33 から 3.08 rad へ飛んだ。そこで、|(g_x, g_z)| ≥ 0.5（ロール 60° 未満）の間は重力角の増分を使い、それ以外はジャイロのピッチ角速度を積分する（`THETA_MIN_PROJECTION`）。
2. **h_to のラッチ**: 仕様書では T3（2 ステップ連続で非接地）の時点の高さを h_to とする。しかしその時点で胴体はすでに約 2·v*·dt（約 0.1 m）上昇しており、H が過小になる（Mid の弾道で H = 0.25 m）。そこで、最初に非接地になったステップの高さを記録し、T3 で h_to としてラッチする。T3 の判定条件は変えていない。変更後の H は 0.29 m。残りの差は、足が地面に触れたまま胴体が上がる分。
3. **apex は1エピソード1回**: 胴体が背中から着地して跳ねると、Air のまま vz の正負の切り替わりが何度も起きる。そのため apex_bonus 用のイベントは最初の1回だけ発火させる。
4. **ω_back と θ の符号**: M1 のとおり `back_pitch_sign = −1` を掛ける。

### 注意

- 背中から着地すると足が接地しないため、Air から抜けない。TM-7（Air の滞在上限）で終了させる必要がある（M3）。
- 足が上がった状態で胴体が地面に触れていても、n_c = 0 なら T3 が成立する。TM-2（胴体接触で失敗終了）で打ち切られる前提とする（M3）。
- 学習中のログ: `Metrics/flip/rate/{air,land,success,success_low,success_mid,success_high}`（トリガーを受理したエピソードだけで平均）と、`Metrics/flip/{triggered,success,max_phase,theta_td,H_max}`。

## M3 の結果（2026-11-14、CPU）

タスク `Mjlab-Backflip-Flat-Unitree-Go2` を登録した（`src/mjlab_backflip/tasks/backflip/backflip_env_cfg.py`）。報酬はまだ終端ペナルティ（−5）だけで、残りは M4 で入れる。テストは全 23 件が通る。

### 観測

- Actor は 223 次元。
  - 1 ステップ 54 次元（gyro 3、acc 3、関節角 12、関節速度 12、前回の行動 12、トルク 12）を 4 ステップ分積む。前回の行動以外には、観測遅れ 1〜2 ステップがかかる。
  - 履歴に含めない項として、g_est 3、θ_est/2π 1、コマンド 3（高さスカラー、トリガー、経過時間）。
  - IMU は、エピソードごとのバイアスとスケール誤差に白色ノイズを加え、±2000 dps、±16 g で飽和させる。
  - g_est と θ_est は、真値に「エピソードごとの初期バイアス + ランダムウォークのドリフト」を足したもの。
- Critic は 115 次元。
  - Actor の 1 ステップ分（ノイズなし）61 次元。
  - 基底の速度 6 次元。
  - フリップの状態 17 次元（h、θ、相の one-hot 6、相の経過時間、τ_air、h_to、H、v*、t_f*、f_flip、トリガー、Δh*）。
  - 足の接地、接地力、足先高さ 12 次元。
  - DR パラメータ 19 次元（摩擦、付加質量、重心 3、Kp 倍率、Kd 倍率、エンコーダーバイアス 12）。仕様書の「約 8 個」より多いのは、エンコーダーバイアスを関節ごとに渡しているため。
- 仕様書にない仮値: 加速度ノイズ ±0.2 m/s²、トルクノイズ ±0.5 Nm、ジャイロバイアス ±0.02 rad/s、加速度バイアス ±0.2 m/s²、IMU スケール誤差 ±2%。

### 終了条件

- TM-2〜TM-8 は失敗（terminated）、TM-1 と TM-9 は打ち切り（time_out）。
- `tests/test_terminations.py` で、それぞれの状況を人工的に作り、対応する項だけが発火することを確かめた。
  - 背中から落とす → TM-2
  - 前に回す → TM-4
  - 速く回しすぎる → TM-5
  - しゃがまない → TM-7
  - 回さずに跳ばす → TM-6（L0 では無効）
  - 補助付きで成功させる → TM-9 だけ

### 見つけた問題と対処

1. **リセット直後に 8 cm 跳ねる（mjlab 1.6.0 の挙動）**
   - 行動遅れ（アクチュエータの遅延バッファ）を入れると、リセット直後の 4 物理ステップは関節目標が 0 rad（脚がまっすぐ）になり、ロボットが跳ねる。
   - 原因: リセット後、行動を適用する前に `scene.write_data_to_sim()` が呼ばれる。そのため、リセットしたバッファに最初に入る値（履歴全体に複製される）が、古い関節目標になる。初回は 0、自動リセットでは前エピソードの最後の目標。
   - 対処: 最後のリセットイベント `hold_default_joint_targets` で、関節目標を既定姿勢（から encoder bias を引いた値）に設定する。自動リセット後も既定姿勢から始まることをテストで確認した。
2. **Critic の DR パラメータが読めていなかった**
   - `SceneEntityCfg` の id はエンティティ内の番号で、`env.sim.model` のフィールドには `asset.indexing.*_ids` で変換した通し番号が要る。変換後、環境ごとに値がばらつくことをテストで確認した。

### 学習前に分かったこと

- ランダムな方策ではトリガーが受理されない。EV-6 のインターロックが、関節速度 1 rad/s 未満を 0.3 s 続けることを要求するため。方策はまず静止して立つことを覚えないと、フリップの報酬を一度も経験しない。M4 の Stand の報酬と、M5 の初期設定で考慮する。
- 制御なしで投げた胴体は、脚が伸びているぶん回転が遅い（与えた 11.8 rad/s に対して約 7.5 rad/s）。Mid の ω* で投げても約 270° で着地する。空中で脚を畳む（tuck）効果が大きい可能性がある。
- train コマンドは `init_at_random_ep_len=True` で走るため、学習開始直後のエピソードは途中から始まる（平均エピソード長 44 ステップ）。トリガーの時刻はエピソード時刻基準なので、最初の数エピソードだけ時刻がずれる。

## M4 の結果（2026-11-14、CPU）

仕様書の報酬表をすべて実装した（`src/mjlab_backflip/tasks/backflip/mdp/rewards.py`、重みは `constants.py` の `REWARD_WEIGHTS`）。テストは全 27 件が通る。

### 実装の規約

- 関数は非負の大きさを返し、符号は重みで表す。`tests/test_rewards.py` で、重みの符号が仕様書の表と一致することと、投げたフリップ（着地補助つき）の全ステップで生の値が負にならないことを確認した。
- 相の切り替えは `phase_gated(env, phases, fn, **kw)`。クラス型の項（feet_distance、tuck）は自分で相を判定する。
- 疎の項（apex_bonus、success_bonus）は、発火したステップで `1/step_dt` を返す。mjlab が dt 倍するので、エピソード合計はちょうど重みになる。1 エピソードに 1 回だけ発火することと、success_bonus の合計が 20 になることをテストで確認した。
- 重みはすべて `constants.py` にある。CLI からは `--env.rewards.<name>.weight` で変えられる。

### 仕様書から変えた点・補った点

1. **Land の項は Done 相にも適用する。** 成功後も 0.5 s は立ち続けるので（EV-12）、その間も直立と静止を評価する。
2. **leg_contact（−1、太ももとすねの接地数）を追加した。** 仕様書の TM-2 の注記「太ももとすねの接触は終了せず減点のみ」に対応する項。重みは仮値。
3. **tuck と land_prep は既定で無効（重み 0）にした。** 仕様書で「任意」の項。仕様書の重み（−1、+3）は `TUCK_WEIGHT_SPEC` と `LAND_PREP_WEIGHT_SPEC` に残してある。M3 で、脚を伸ばしたままでは回転が遅いことが分かったので、学習の様子を見て有効にする候補とする。q_tuck は既定姿勢から thigh +0.6、calf −0.7 rad（仮値）。
4. **feet_distance の基準幅は、最初の呼び出し時（リセット直後の立ち姿勢）の値を使う。**
5. **rear_before_front は「後脚が浮いている割合 × 前脚のどれかが接地している」とした。** 仕様書の「前足より先に後ろ足が離れる」を、ステップごとの密な減点として書いたもの。

### 記録（EV-13、EV-14）

トリガーを受理したエピソードだけで平均し、`Metrics/flip/` に出す。
- `rate/{air,land,success,success_low,success_mid,success_high}`
- `ev14/height_err`: |H_max − Δh*|、Air に到達したエピソード
- `ev14/theta_td_err`: |θ_td − 2π|、Land に到達したエピソード
- `ev14/land_acc_peak`: Land 中の IMU 加速度ノルムの最大値
- `ev14/torque_peak`、`ev14/joint_vel_peak`: トリガー後の最大値
- `ev14/xy_drift`、`ev14/yaw_change`: エピソード開始からのずれ
- `phase_time/<相>`: 各相に滞在した時間

相ごとの累積報酬は、項名が相ごとに分かれているので、`Episode_Reward/<項>` で読める。

### 学習なしの確認

- CPU で 64 env × 3 イテレーションを回し、すべての減点項の `Episode_Reward` が 0 以下になることを確認した。
- ランダムな方策ではトリガーが受理されないため、フリップの項はすべて 0 のまま（M3 の注意のとおり）。

## M5 の結果（2026-11-14、CPU）

カリキュラム `flip_curriculum`（`src/mjlab_backflip/tasks/backflip/mdp/curriculum.py`、設定は `constants.CURRICULUM`）を実装した。テストは全 32 件が通る。

### 動作

- ステージと終了レベルは S1(L0) → S2(L0) → S3(L1) → S4(L2) の 1 本の段階として進める。
- 進級の指標（トリガーを受理した、終了済みのエピソードの EMA。時定数は約 2000 エピソード）:

  | 遷移 | 指標 | ρ |
  |---|---|---|
  | S1 → S2 | 回転到達率（Land に到達し、θ_td ≥ 2π − 現レベルの成功判定マージン） | 0.4 |
  | S2 → S3 | Mid の成功率 | 0.4 |
  | S3 → S4 | コマンド別成功率の最小値 | 0.7 |

- 方策のイテレーション（24 env ステップ）ごとに判定する。
  - ρ 以上が 100 回続いたら進級する。0.5ρ 未満が 200 回続いたら降格する。
  - どちらも、そのステージに 500 イテレーション以上いることが条件。
  - ステージが変わると EMA と連続回数をリセットする。
- ステージごとに切り替えるもの
  - コマンドの集合と終了レベル（`FlipCommand.set_stage`）。
  - Land の 6 項の重み: S1 は 0、S2 以降は仕様書の値。
  - success_bonus: S1 は 0、S2 で 1000 イテレーションかけて 0 → 20 へ線形に上げる、S3 以降は 20。
  - DR の範囲: S1〜S2 は狭い範囲（`DR_NARROW`）、S3 以降は仕様書の範囲（`DR_FULL`）。
- カリキュラムはリセット処理の最初に呼ばれる（コマンドとイベントのリセットより前）。そのため、終わったエピソードのフリップ状態を読めて、DR の変更も同じリセットから効く。
- ログ: `Curriculum/flip/{stage,level,metric,rotation_rate,success_rate,success_low,success_mid,success_high,above_streak,below_streak,success_bonus_weight}`。stage は 1〜4 で S1〜S4 を表す。

### CLI

- `--env.curriculum.flip.params.fixed-stage <0..3>`: ステージを固定する（デバッグ用）。−1 で自動。
- `--env.curriculum.flip.params.start-stage <0..3>`: 開始ステージ。
- `--env.curriculum.flip.params.{thresholds,promote-iters,demote-iters,min-iters,success-ramp-iters,ema-episodes}`
- 注意: tyro は既定値から CLI の型を決める。既定値を `None` にすると、そのフラグは `None` 以外を受け付けない。`fixed_stage` は当初 `None` で値を渡せなかったため、整数の −1 を「自動」の意味に変えた。

### 注意

- ステージはチェックポイントに保存されない（mjlab が保存するのは `common_step_counter` だけ）。`--agent.resume` で再開するときは `start-stage` で再開するステージを指定する。
- `num_steps_per_env` は runner の設定（24）と合わせる必要がある。
- S1→S2 の ρ = 0.4 は仕様書にない仮値。S4 の到達条件（最終的な成功率の目標）は仕様書でも未決のまま。
- 実際の進級は学習しないと確かめられない。CPU では、合成したエピソード結果で進級・降格・ランプ・DR 切り替えを確認し（`tests/test_curriculum.py`）、train コマンドで `Curriculum/flip/*` が記録されることと、`fixed-stage 2` で S3（L1、success_bonus 20）になることを確認した。

## 最初の学習の結果と修正（2026-11-14、Colab T4、2048 env、約 500 イテレーション）

ログ: `2026-11-14_05-49-16_colab`

### 起きたこと

方策が「すぐ倒れてエピソードを終わらせる」ことを覚えた。

- 平均エピソード長は 10〜30 ステップ（最大 250）で、ほぼすべてが TM-2（胴体接触）で終わった。time_out は 0。
- model_250 を動かすと、リセットから 0.32 s で自ら膝を曲げ、胴体を接地させた。
- `Policy/mean_std` は 0.087 まで縮み、この行動に収束した。ステージは S1 のまま。

### 原因

1. **終端ペナルティが −0.1 にしかなっていなかった。** mjlab は報酬を dt（0.02 s）倍するため、`mdp.is_terminated`（重み −5）では失敗 1 回あたり −5 × 0.02 = −0.1 になる。ログの `Episode_Reward/terminal_penalty` = −0.02（= −0.1 ÷ 5 s）と一致する。疎のボーナスには入れていた dt の補正を、終端ペナルティには入れていなかった。
2. **Stand の項がすべて減点だった（仕様書の設計）。** 学習初期の方策では、5 s 立ち続けると数点の減点になる。−0.1 を払って倒れるほうが得点が高い。

### 修正（仕様書も改訂）

- `terminal_penalty` は `terminated_once`: 失敗したステップで `1/dt` を返し、1 回で −5 にする。打ち切り（time_out、成功後）では発火しない。
- 生存ボーナス `alive`（+1/s、`mdp.rewards.alive`）を共通項に追加した。失敗していない間だけ入る。
- EV-14 の `xy_drift` と `yaw_change` のバグを直した。`FlipCommand.reset()` はシーンのリセット後に呼ばれるので、ロボットの現在の姿勢は既に新しいエピソードのものになっている。エピソード最後の姿勢を毎ステップ `last_xy` と `last_yaw` に保存し、それを使う。

### ログの読み方（mjlab 1.6.0）

- `Episode_Termination/*` は割合ではなく件数。直前のリセットで終わったエピソードのうち、その条件で終了した数を表す。

## 2 回目の学習の結果と修正（2026-11-14、Colab T4、2048 env、約 400 イテレーション）

ログ: `2026-11-14_06-25-24_colab`

### 起きたこと

- 生存ボーナスの導入で、方策は 5 s 立ち続けるようになった。平均エピソード長は 250 で、ほぼすべてが time_out で終わる。`Train/mean_reward` は約 4.6 で頭打ちになった（alive の +5 から立っている間の小さな減点を引いた値）。
- トリガーを受理したエピソードは 0.5 %（`Metrics/flip/triggered` = 0.005）だった。そのため Crouch 以降の報酬は一度も入らず、ステージは S1 のまま。

### 原因

EV-6 の静止判定を、全関節の速度の最大値と胴体の角速度について、毎ステップの瞬時値で行っていた。学習中は行動に探索ノイズ（std 0.14）が乗るので、関節が細かく震えて条件が途切れる。

model_250 を CPU で確率的に動かし、立っている間の軌跡で判定方法を比べた（128 env、2.5 s）。

| 判定 | 2.5 s 以内の受理 | 待ち（中央値） |
|---|---|---|
| 瞬時値: 関節速度の最大値 < 1.0、かつ角速度 < 0.5（元の実装） | 0 % | — |
| 瞬時値: 関節速度の最大値 < 2.0 | 2 % | 1.8 s |
| ローパス 0.1 s: 関節速度の最大値 < 1.0 | 25 % | 1.1 s |
| ローパス 0.1 s: 関節速度の平均 < 1.0（採用） | 95 % | 0.64 s |

### 修正（仕様書の EV-6 も改訂）

- 時定数 0.1 s の一次ローパスをかけた「胴体の角速度ノルム < 0.5 rad/s」と「12 関節の速度の絶対値の平均 < 1.0 rad/s」が、0.3 s 続いたら受理する（`FlipCommand.filt_ang_vel`、`filt_joint_vel`、定数は `STABLE_*`）。しきい値は変えていない。
- テスト `test_interlock_tolerates_exploration_noise`: 制御なしで既定姿勢の周りに std 0.14 の行動ノイズを入れると、2 s 以内の受理は元の判定で 0 %、新しい判定で約 70 %。

### 測り方の注意

学習途中の方策を学習時の設定で動かすと、トリガーを受理したあと失敗して、すぐリセットされる環境がある。そのため、ある時刻に「受理済みの環境の割合」を数えると低く出る（3 s 時点で約 35 %）。未受理の環境はリセット直後（エピソード長 12〜38 ステップ）で、次のトリガー時刻に達していないだけだった。

## 3 回目の学習の結果と修正（2026-11-14、Colab T4、2048 env、約 270 イテレーション）

ログ: `2026-11-14_06-59-39_colab`

### 起きたこと

- 方策は 5 s 立ち続け、`Train/mean_reward` は約 4.0 で頭打ちになった。トリガーの受理は 1 % 未満（`Metrics/flip/triggered` ≈ 0.006）のまま。
- 受理された少数のエピソードでは、`rate/air` と `rate/land` が 0.5〜1、`torque_peak` が 45 Nm だった。跳んで着地するところまでは進んでいる。

### 原因

model_250 で、64 env × 3 s のトリガー受理を比べた。

| 方策 | 胴体角速度（フィルタ後）の中央値 | 受理 |
|---|---|---|
| 平均の行動 | 0.12 rad/s | 94 % |
| 確率的な行動（std 0.27） | 0.66 rad/s | 2 % |

1. RSL-RL の行動ノイズはスカラーで、状態によらない。フリップの探索で std が下がらない限り、Stand でも胴体が揺れる。前回は 250 イテレーションで std 0.14 まで下がったが、今回は 0.27 だった。
2. 受理されなければ、生存ボーナス +5 を確実に得られ、失敗の −5 もない。受理の可否が方策の揺れで決まるので、受理を避けることが最適になる（報酬の抜け穴）。

### 修正（仕様書の EV-6 も改訂）

- `FlipCommandCfg.interlock`（既定 True）と `FlipCommand.interlock` を追加した。オフのときは、トリガー入力の時刻で必ず受理する。
- カリキュラムの `interlock_from_stage`（既定 3 = S4）より前のステージでオフにし、S4 でオンにする。`Curriculum/flip/interlock` に記録する。
- play 設定（カリキュラムなし）と `record_video.py` では常にオン。
- 確認: 同じ model_250 を学習時の設定（S1）で確率的に動かすと、3 s 以内の受理は 2 % から 88 % になった。残りはリセット直後の env。
