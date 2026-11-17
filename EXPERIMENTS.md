# 实验记录：实测数字、失败分析与当前阻塞点

本文件只写**实际跑出来的结果**。计划、预期和推测分开标注。原始结果在 `runs/summary/results.json`（精简版，进仓库）与 `runs/eval/*.json`（完整 trace，不进仓库）。

更新时间：2026-11-17。

---

## 1. 目标与实测对照

项目目标（用户设定）：

| # | 目标 | 实测 | 状态 |
|---|---|---|---|
| 1 | 用官方 collect 的数据训练，sensor 只用 wrist | 1000 次尝试 → 569 条成功示范；两只相机都存，**训练只用 `hand_camera`** | ✅ 完成 |
| 2 | A/B/C 三个 skill 各自 SR ≥ 90% | A 71–77%、B 20%、C 未测（专家上限 68.7%） | ❌ 未达标 |
| 3 | Foresight RL 把 B 微调成 B′；期望 BC≈90%、ABC≈50%、B′C≈80%、AB′C<ABC（AB′ 退化） | 代码有骨架，**实验一次都没跑**；前置条件均不满足 | ❌ 未开始 |

---

## 2. 数据

`data/formal_rgb`：

| 项目 | 数值 |
|---|---|
| 采集尝试 | 1000（独立 seed 0–999） |
| 成功示范 | **569** |
| 划分（按 seed 分组，同 seed 不跨集） | train 457 / val 56 / test 56 |
| 训练帧数 A / B / C | 42063 / 21616 / 18879 |
| 全部帧数 A / B / C | 52398 / 26892 / 23442 |
| episode 长度 | 均值 180.5 步，140–256 |
| 相机 | `hand_camera` + `base_camera`，128×128，uint8 |
| 审计 | 0 重复、0 相同动作序列 |

### 2.1 阶段边界可复现性（全量 569 条，`charger/validate_shards`）

单次正演（reset 一次后只 step，不 `set_state_dict`）：

| 结果 | 数量 |
|---|---|
| A/B/C 边界与录制完全一致 | 387 |
| 边界偏移（A 2 / B 0 / C 5） | 7 |
| 只在 C 失败 | 175 |

**A 边界 2 条不符、B 边界 0 条不符** → A、B 的数据与标签一致，训练 A/B 不受影响。

### 2.2 C 的问题（真实原因）

`terminated` 在**首次成功那一步**为真（569/569 与录制 C 边界精确对齐），但源录制在成功后又多记了 1–8 步（均值 4.7，**569/569 都有**）。这些尾部帧不是同一技能的合法延续。

更关键的是：**专家动作本身复现 C 的上限只有 68.7%**。

| 检查 | 结果 |
|---|---|
| 专家动作逐帧回放，能否复现录制 C | **391/569 = 68.7%** |
| 失败那批的最终距离 | 中位数 **4.5 cm**，达到 ≤5 mm 的：**0 条** |

对比 A：**专家 A 段回放 569/569 成功，且抓取步号与录制完全一致。**

结论：C 的 90% 目标在这份数据上**数学上不可达**（上限 68.7%）。这不是标签 bug，是细小接触任务在二次回放时接触状态不可重建。

---

## 3. 训练

| 运行 | 技能 | 步数 | 最佳验证 loss | 耗时 |
|---|---|---|---|---|
| `runs/formal/A` | A 单视角 | 12000 | 0.00703（最后一步） | 9 分钟 |
| `runs/formal/B` | B 单视角 | 12000 | 0.00767（最后一步） | 11 分钟 |
| `runs/dual/A` | A 双视角 | 12000 | 0.00742（最后一步） | 13 分钟 |
| `runs/formal_labels_invalid/C` | C（标签无效，故意隔离） | 12000 | 0.00781（step 9000） | 11 分钟 |

统一配置：`seed=0`、batch 64、AMP、`--proprio --relative-actions`、`hand_camera`。

**注意：`runs/formal/` 下故意没有 C。** C 的标签问题没解决前，把它放进去会让链式评估悄悄拿到无效检查点。

---

## 4. 真实 rollout 评估

所有数字来自 `charger/evaluate`，100 个全新 seed（10000–10099）或 50 条 held-out 专家前缀，budget 200/192，4 步 replanning。

| 链 | 成功 | 成功率 | Wilson 95% | 备注 |
|---|---|---|---|---|
| A（单视角） | 71/100 | 0.71 | [0.615, 0.790] | |
| A（单视角，同配置重跑） | 76/100 | 0.76 | [0.668, 0.833] | **可复现性问题** |
| A（双视角 wrist+base） | 62/100 | 0.62 | [0.522, 0.709] | 加了全局相机**更差** |
| B（专家 A 前缀起步） | 10/50 | 0.20 | [0.112, 0.330] | |
| AB 链（不 reset） | 4/100 | 0.04 | [0.016, 0.098] | A 77 → B 只接住 4 |
| C / BC / ABC | — | — | — | **从未评估** |

### 4.1 A 为什么失败

- **不是可执行性问题**：专家 A 段 569/569 可复现。
- **不是卡死**：失败 rollout 全程在动。
- **真实形态**：TCP 停在充电器上方 5–15 cm 下不去，或贴到侧面 ~1 cm 却没夹住。属于**末端抓取定位精度**问题。
- **多视角无效**：双视角 62% < 单视角 71–77%，已用真实对照实验证伪“补视角就能补上信息”的假设。

### 4.2 B 为什么失败

从专家 A 前缀出发，6 条样本：

- 最小距离能到 **0.9–3.5 cm**，但多数在 41–80 mm 处停住、姿态误差 24–42°；
- 少数接近到 1 cm 内但姿态超 10° 阈值；
- 失败模式是**进入 1 cm 前停止收敛并漂走**，不是走错方向。

### 4.3 诊断视频（已提交）

`runs/videos_A/`、`runs/videos_B/`、`runs/videos_compare_A/` 共 14 个 mp4。每个视频上面是腕部相机、下面是全局相机，每帧叠加当前步数、到目标点的距离、姿态角和是否抓持。生成命令：

```bash
python -m charger.record_video --mode A --seeds 10000 10001 10006 --out runs/videos_A --scale 3
python -m charger.record_video --mode B --seeds 0 205 --data data/formal_rgb --out runs/videos_B --scale 3
python -m charger.compare_video --mode A --seeds 10000 10001 --out runs/videos_compare_A
```

逐帧数据在 `runs/videos_*/**/*_report.json`。视频中可直接看到的两点：

- **A 失败**：夹爪全程没有闭合到抓持（`grasp 0`），末端停在充电器上方或侧面。
- **B 失败**：最小距离能到 1.4–2.2 cm（接近 1 cm 门槛），随后**距离重新变大**到 2.8–4.2 cm。

录制过程中发现并修正了一个真实缺陷：`record_video` 的第一版没有回放专家 A 前缀，B 策略从静止状态起步，`ever_grasped=False`、全部失败。修正后 B 出现成功样本（seed 205）。**任何 B/C 评估如果从 plain reset 开始都是无效的**，这条协议不能省。

B|A 只有 5.2%（4/77），是当前全链的瓶颈。

---

## 5. 已知问题：评估不可完全复现

同配置、同检查点、同 seed 重跑，A 段结果只有 **82–83/100** 一致。

已排除：

- 权重相同（`best.pt` 与 `last.pt` 的 EMA 权重逐张量相同）；
- 模型对固定输入**逐位确定**（`cudnn.benchmark` True/False 结果相同）；
- 物理重放逐位确定；
- 每次都 `torch.manual_seed` 过。

已定位的入口：同一 seed 前 8–10 步**完全相同**，之后以 1e-8 m 量级开始分叉，此时手臂离物体还有 12–26 cm、没有接触；进程内连续跑多条 episode 时差异从第 35–47 步进入。用 `charger.repro` 在**两个独立进程**各重放同一 seed：21/46 次 replan 的输入输出逐位相同。

**结论**：单条 rollout 可复现，但长进程内环境存在历史耦合（最可能是 CPU 物理端的缓存/线程调度，尚未定位到具体代码）。任何“差 3–5 个点”的结论都不可信，正式对比必须多 policy seed 并报方差。

---

## 6. 被证伪的中间结论（避免重蹈覆辙）

这些说法在过程中出现过，后来被实测推翻，记录在此防止再次引用：

| 曾经的结论 | 实测结果 |
|---|---|
| “渲染/取观测会改变物理，导致 C 失败” | 假。同进程内四种 obs/render 组合距离序列逐位相同 |
| “C 的问题是多记了 1–8 帧尾巴，截断即可” | 不够。专家回放上限 68.7%，失败距离 4.5 cm，与尾巴无关 |
| “A 可执行性只有 25%” | 假。那是把 C 的失败混进来；A 是 569/569 |
| “A 失败是因为夹爪朝向反了 180°” | 假。欧拉角分解假象；真实原因是末端定位精度 |
| “多视角能补上 wrist 缺失的信息” | 假。双视角 62% < 单视角 71–77% |

---

## 7. 下一步（按优先级）

1. **确定 C 的处理方式**：专家上限 68.7%，要么重新设计采集（更小控制步长、成功帧精确终止、或从确定状态分技能采集），要么把 C 的验收目标改为“接近专家上限”。
2. **A 的末端精度**：当前 71–77%。需要改的是数据规模/抓取多样性或动作表示（如末端位姿增量），不是加视角或加步数。
3. **评估方差**：查清长进程内环境的数值耦合，或在正式对比中强制“一进程一条 episode”并报方差。
4. **B 的收敛**：A 达标后再处理，当前 B 在 1 cm 外就停住。
5. **Foresight / B′**：必须等 A、B、C 各自达标。当前 ABC 只有 4%，任何 B′ 增益都会被淹没。

---

## 8. 复现本文件所有数字

```bash
# 数据校验（573 秒，4 分片）
python -m charger.validate_shards --data data/formal_rgb --out runs/validation/physics_all.json --shards 4

# 训练
python -m charger.cache_frames --data data/formal_rgb --stage A --camera hand_camera --out data/frame_cache
python -m charger.train_forever --stage A --steps 12000 --batch-size 64 --frame-cache data/frame_cache --out runs/formal
python -m charger.train_forever --stage B --steps 12000 --batch-size 64 --frame-cache data/frame_cache --out runs/formal

# 评估
python -m charger.evaluate --mode A --episodes 100 --budget 200 --checkpoints runs/formal --out runs/eval/A_100.json
python -m charger.evaluate --mode B --episodes 50 --budget 200 --checkpoints runs/formal --data data/formal_rgb --out runs/eval/B_50.json
python -m charger.evaluate --mode AB --episodes 100 --budget 192 --checkpoints runs/formal --data data/formal_rgb --out runs/eval/AB_192.json

# 汇总
python -m charger.summarize --runs runs --out runs/summary/results.json
```
