---
name: tailor-report-interpretation
description: 解读 lerobot-tailor 的数据集验收报告。触发：拿到 tailor check 的终端输出 / --json 报告 / markdown 报告，或被问「这个 FAIL 什么意思」「REJECTED 的数据还能用吗」「要不要管这条 WARN」。输出逐条严重度判定（必修 / 可豁免 / 观察即可）加行动建议。
version: 0.1.0
license: MIT
---

# 解读 tailor 验收报告

## 你会拿到什么

三种载体，内容同源：

- 终端表格：每行 `检查名 | 状态 | 摘要`，结尾一行 verdict（ACCEPTED / ACCEPTED WITH WARNINGS / REJECTED）
- `--json`（机器可读）：`{verdict, checks: [{name, status, message, details}]}`
- `--markdown`（PR / dataset card 用）

状态四种：PASS / WARN / FAIL / SKIP。**SKIP 不参与判定**——一条 SKIP 意味着「本次没查」，不是「没问题」（见「假绿陷阱」）。

## 总原则（先说结论的读法）

1. **REJECTED ≠ 数据不可用**。REJECTED 只说明「至少一条硬检查没过」。结构级 FAIL（文件缺失、指针断裂）通常真不能用；语义级 FAIL（内容可疑类）有相当比例是场景特性，逐条分诊后才能下结论。
2. **先分桶，再逐条**。25 条检查分四个桶，桶决定默认态度：

| 桶 | 检查 | FAIL 时默认态度 |
|---|---|---|
| 结构级 | episode_files、video_files、total_frames、schema_consistency、frame 完整性类 | 必修——数据物理上缺东西或指针断裂，训练管线会直接崩 |
| 语义级 | action_quality（frozen/clipped/jump，均报 WARN 不触发 REJECTED）、zero_variance、duplicate_episodes、video_content、video_sync、video_black | 查场景库——相当一部分是数据集固有特性，不是缺陷 |
| 统计级 | stats_recompute、timestamps、episode_lengths、fps | 多数良性，看幅度 |
| 信息级 | portability、datasheet/provenance | 记录即可，不影响训练 |

3. **语义级的判定权在数据语义，不在数值本身**。同一条「贴边界」，连续关节维度是限幅告警，二值 gripper 维度是正常工作点。

## 技术性命中场景库（误报与豁免）

逐条给「现象 → 为什么是误报 → 怎么确认 → 怎么处置」。命中报告条目时按库对照，库里没有的场景不要硬套。

### 1. 二值维度的 range edge pegging（最高频）

**现象**：action_quality（WARN 级）报 "range-edge pegging in N episode(s)"，N 接近全部 episode。

**为什么是误报**：贴边界判据是「某维度在 stats.json 声明范围的 1% 邻域内持续 ≥5 帧」。gripper 的动作命令是二值（0=开, 1=关），而 stats 的 min/max 恰好就是 [0, 1]——1% 邻域 `[0, 0.01]` 和 `[0.99, 1]` 把仅有的两个合法取值全部包住。**gripper 每一次持续抓握都满足「持续贴边」**，这是该判据在离散维度上的必然命中，不是限幅。

**确认方法**：取 action 数组，看命中维度的取值集合——`len(np.unique(col)) <= 2` 即二值；或查 info.json 的 features（action.names 里含 gripper 字样的维度）。一行验证：

```python
import numpy as np
arr = snap.col_array(ep_idx, "action")  # (n_frames, dims)
[np.unique(arr[:, d]).tolist() for d in range(arr.shape[1])]
```

**处置**：二值维度豁免，此条不构成拒收理由。真正要警惕的是**连续维度**（关节位置/速度）的持续贴边——那是限幅、标定错或控制器饱和的信号。注意当前检查按 episode 汇总输出、不按维度拆分：如果同一个 FAIL 里既有 gripper 维度又有连续维度命中，不能整体豁免，需要用上面的方法分辨是哪个维度贡献的。

### 2. zero_variance 命中「从未动过的传感器」

**现象**：某 state 维度全程恒值。

**为什么常见**：硬件没接该传感器（读数恒 0），或该自由度在全部任务中确实没动过（锁死的关节）。

**确认**：恒值是 0 还是别的数。恒 0 多半是没接；恒非零可能是锁位。

**处置**：训练前应显式 drop 该维度（否则 BN/归一化除零由框架兜底，行为不可预期）。数据本身可用，但这个维度是死列。

### 3. frozen_action 命中任务间过渡

**现象**：episode 内有 ≥10 帧动作完全相同。

**为什么常见**：机械臂在任务段之间静止等待、示教者在思考、reset 动作前的停顿。人遥控（teleop）数据尤其普遍。

**处置**：看命中 episode 的占比。零星命中且时长 <1s → 观察即可；连续整段冻结才是示教中断 / 录制卡死的信号。`frozen_frames` 阈值可在 profile 调。

### 4. stats_recompute 在采样模式下的天然偏差

**现象**：--max-episodes 采样跑出 drift 超阈值。

**为什么必然**：重算用的是**采样子集**，stats.json 声明的是**全量统计**。子集均值天然偏离全量均值；last 采样（默认）对时间上漂移的数据集偏差更明显。这不是数据坏了，是两次统计的样本不同。

**处置**：drift 幅度与采样比例一起看——采样比 <20% 时 1-2% 的 drift 属预期；要下结论请全量跑（去掉 --max-episodes）再判定。**不要用采样模式的 stats_recompute 结果拒收数据集**。

### 5. timestamps spacing 命中遥控节奏

**现象**：帧间距不均匀。

**为什么常见**：teleop 数据按操作者节奏采集，fps 是名义值不是硬约束。传感器丢帧也会造成。

**处置**：偏差 <10% 且无长洞 → 观察即可；出现 >1s 的洞要查录制链路（这是丢真帧）。下游用 timestamps 重采样的话此项要认真对待。

### 6. duplicate_episodes 命中刻意重复

**现象**：两个 episode 的 state+action 轨迹字节级相同。

**为什么常见**：同一演示故意重复采集（数据增广）、reset 动作天然相似。

**处置**：少量重复 → 用 `tailor clean --dedupe` 清掉即可，不构成拒收。大量重复（>10%）要查采集脚本是不是循环写坏了。

### 7. video_files 的 gap WARN（v3 共享分片）

**现象**：shard 内有无人认领的时间段。

**为什么设计成 WARN**：删除 episode 会合法地在共享文件里留下空洞（文件不重写）。这是 0.11.0 的设计行为。

**处置**：确认 gap 位置恰好对应已删除的 episode 区间 → 忽略。gap 指向仍存在的 episode → 指针错位，升级处理。

### 8. video_content 的 mid-file black（腕部相机）

**现象**：视频中段短暂全黑。

**分界**：check 已内置「黑前亮度」判据——从亮场景**突变**到黑才报（硬件故障形态）。渐暗到黑（物体贴上镜头）已被过滤。仍报出的条目，先看是哪路相机：腕部相机短暂遮挡即便漏过判据，也多为操作特性。

**处置**：头部相机（全景）中段黑 → 认真查（镜头盖、线缆）；腕部相机 <2s 的黑段 → 观察即可。

### 9. episode_lengths 命中短集

**现象**：个别 episode 远短于中位数。

**为什么常见**：失败演示、提前终止的尝试——很多团队**故意保留**失败数据。

**处置**：问数据集作者（或读 dataset card）失败集是否 by design。是 → 豁免；不是 → 查录制中断。

## 假绿陷阱（SKIP 语义）

远程 HF 数据集默认 meta-only 拉取：episode_files / video_files / video_black / video_content / video_streams 全部 SKIP，verdict 照样 ACCEPTED、退出码 0。**这不是全量通过，是没查**。CI 门禁要么加 `--full`（拉文件），要么加 `--fail-on-skip`（有 SKIP 时退出码 3）。

分诊时先数 SKIP 条目：结构级检查有 SKIP → 本次报告对「文件是否齐全」**没有发言权**，不要在汇报里写「文件完整」。

## 行动模板

拿到报告后按此顺序产出结论：

1. **数桶**：FAIL 几条、各属什么桶；SKIP 几条、是否影响结论完整性
2. **结构级逐条列「必修」**：附 message 里的具体清单（缺哪些文件、哪个指针断）
3. **语义级逐条过场景库**：命中的写明「对应场景 N，豁免理由，确认证据」；库外的新场景不要擅自豁免，列为「待人工判定」
4. **统计级看幅度**：给出数字和采样比例，不贴标签
5. **输出三栏结论**：必修（修什么）/ 豁免（为什么）/ 观察（下次复查什么）

给用户的最终表述里，把「REJECTED 但全部命中场景库」和「REJECTED 且有结构级 FAIL」明确区分开——前者是「检查器需要学习你的数据集特性」，后者是「数据真的坏了」。
