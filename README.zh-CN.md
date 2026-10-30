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

退出码让它可以直接当 CI 门禁用：**0** 收货 · **1** 拒收 · **2** 不可读。
`--json` 输出机器可读结果；`--max-episodes N` 对大数据集只抽查前 N 个 episode。

## 11 条验收检查（v0.1）

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

已在一个真实的 921 MB 分发数据集（`lerobot/unitreeh1_fold_clothes`，38 episodes / 1.9 万帧 / 双目相机）上验证，配合注入坏数据的确定性合成 fixture——每种坏数据都被至少一条测试抓住，两种格式版本各验一遍。

## 状态

v0.1 验收套件已交付。路线图：filter/clean 命令、YAML 配方、视频完整性检查、PyPI 发布。计划见 Issues。
