# tailor

**把分发来的机器人数据集精炼为可直接训练的数据。**

[English](README.md) | 简体中文

tailor 是面向机器人学习数据集（LeRobot **v2.0 / v2.1 / v3.0**）的数据精炼流水线：

- **Accept（验收）**——对分发来的数据集做入库检验：格式版本合规、元数据与数据一致（"货对不对板"）、传输完整性、维度与统计快照对齐训练配置
- **Cut（裁剪）**——episode 级过滤（帧级问题升级为整条剔除，保住 action chunk 的连续性）
- **Sew（缝制）**——只做确定性清洗：时戳重建、统计重算、视频重编码。不做插值补帧
- **Re-check（复检）**——清洗产物必须重新通过整套验收才算出厂
- **Register（登记）**——血缘台账：源版本、剔除了哪些 episode 及原因、配方 hash

配方用声明式 YAML 存进 git；每个数据集可复现、可审计。

*tailor-made datasets for your training.*（为你的训练量身裁衣）

## 快速上手

```bash
pip install tailor            # 发布后可用；当前可先：
pip install git+https://github.com/XBR-1111/tailor

tailor check /path/to/lerobot/dataset
```

```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ check             ┃ status ┃ detail       ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━┩
│ info_json         │  PASS  │              │
│ codebase_version  │  PASS  │ v3.0         │
│ fps               │  PASS  │ 50           │
│ episodes_metadata │  PASS  │ 38 episodes  │
│ episode_files     │  PASS  │              │
│ total_frames      │  PASS  │ 19000        │
│ episode_lengths   │  PASS  │              │
│ task_indices      │  PASS  │ 1 tasks      │
│ state_action_dims │  PASS  │              │
│ nan_inf           │  PASS  │              │
│ timestamps        │  PASS  │              │
└───────────────────┴────────┴──────────────┘
ACCEPTED  ·  11 passed, 0 warn, 0 failed
```

退出码让它可以直接当 CI 门禁用：**0** 收货 · **1** 拒收 · **2** 不可读（`--fail-on-warn`
让 WARN 也以 1 退出）。`--json` 输出机器可读结果；`--markdown report.md` 写出可贴
PR / dataset card 的报告；`--max-episodes N` 对大数据集只抽查前 N 个 episode。

## 可配置的验收 profile

什么算合格，每个机器人、每个任务、每种训练配置都不一样。profile（YAML
存进 git——就是你的数据 SLA，有版本历史、可审计）按检查项覆盖阈值与
severity；tailor 是执行它的引擎：

```yaml
# my_requirements.yaml
checks:
  codebase_version:
    allowed: [v2.1]     # 本组只收 v2.1
  fps:
    allowed: [30, 50]   # 硬件实际帧率
  episode_lengths:
    min: 100            # policy window 需要余量
    severity: fail      # 短 episode 在这里是硬拒收
  timestamps:
    spacing_tolerance: 0.2
```

```bash
tailor check /data/dataset --profile my_requirements.yaml
```

任意检查项的 WARN/FAIL 可以互相重映射。profile 里出现未知的检查名或
配置键会立刻报错——需求文件里的 typo 不能静默地保留一个你想改掉的默认值。

## 22 条验收检查（v0.1）

| 检查项 | 抓什么问题 |
| --- | --- |
| info_json | meta/info.json 缺失或不是合法 JSON |
| codebase_version | 格式版本不在 v2.0/v2.1/v3.0 支持范围内 |
| fps | fps 非正数或异常偏小 |
| episodes_metadata | episode 元数据缺失或不可解析 |
| episode_files | 数据分片/parquet 在磁盘上缺失（v2 按文件、v3 按分片指针查） |
| total_frames | 声明的总帧数 ≠ 各 episode 长度之和 |
| episode_lengths | 长度为零/负数；过短的 episode（会破坏 chunk 采样） |
| task_indices | task 引用越界；声明的任务数与实际不符 |
| state_action_dims | state/action 维度与 info.json 声明不符 |
| nan_inf | state/action 里出现 NaN/Inf |
| timestamps | 时戳非单调；帧间隔偏离 1/fps 超过 10% |
| stats_recompute | 声明的 stats.json mean/std 与重算值漂移超标 |
| action_quality | frozen（连续相同 action）；某一 action 维持续钉在声明范围边界；超过 k·σ 的突变跳变 |
| video_files | 声明的视频流在磁盘上缺失（v2 按每集文件、v3 按分片指针查） |
| video_streams | 视频不可解码；帧数与 episode 长度不符；分辨率与声明 shape 不符 |
| zero_variance | 永不变化的维度（std < eps）：归一化会除零 |
| schema_consistency | 跨 episode 的列集/dtype 漂移 |
| frame_continuity | 丢帧（行数 vs 声明长度、frame_index 有洞） |
| training_readiness | episode 短于你的 policy chunk_size/horizon（配置后启用） |
| portability | symlink（传输即断）、data/ 里的杂散文件、意外的顶层条目 |
| duplicate_episodes | 跨 episode 的逐字节相同 state/action 轨迹 |
| splits | 声明的各 split episode 区间重叠（train/eval 泄漏） |

已在一个真实的 921 MB 分发数据集（`lerobot/unitreeh1_fold_clothes`，38 episodes / 1.9 万帧 / 双目相机）上验证，配合注入坏数据的确定性合成 fixture——每种坏数据都被至少一条测试抓住，两种格式版本各验一遍。这个真实数据集自身带着两个被 tailor 首跑就抓到的问题：一个 action 维贴边界 29 帧（gripper 夹住叠好的衣服，WARN 级）；以及一处真实的元数据缺陷——`cam_left` 的 episode 分片指针全部指向 `file-000.mp4`，但它的 500 帧实际躺在无人引用的 `file-001.mp4` 里。已用官方读取器验证：`LeRobotDataset` 加载正常，但 episode 37 的所有帧都抛 `IndexError`（视频索引封顶 18500）——训练只要采到最后一个 episode 就崩。

## 状态

v0.1 验收套件已交付。路线图按优先级排列：

**验收纵深（v0.2）**——剩余：

- 按 HF repo id 远程验收（下载前先查元数据）、zip 归档支持

**流水线闭环（v0.3+）**——Cut/Sew/Re-check/Register：

- `filter` / `clean` 命令构建在 lerobot 官方 `dataset_tools`（delete_episodes、compute_stats）之上，挂 `[lerobot]` extra——产物天然保持格式兼容；验收本体保持零重依赖
- YAML 配方（跟随上游 `recipe.py` 的方向，不自造 schema）
- 血缘台账 + 登记

生态参照：lerobot-doctor（诊断，已进官方 dataset visualizer）、Calibra（episode 处置建议）。tailor 的车道是**验收门禁 + 可执行的流水线**——verdict + 退出码当 CI 门，然后是带血缘的确定性精炼，不做又一份报告。
