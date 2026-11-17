# PlugCharger 视觉技能实验

在 ManiSkill 3 `PlugCharger-v1` 上，用官方 motion-planning 示范训练三个视觉技能（A 抓取 / B 对齐 / C 插入），并（计划）用 Foresight Residual RL 把 B 微调成 B′。

**先看 [`EXPERIMENTS.md`](EXPERIMENTS.md)：那里有全部实测数字、失败原因和当前阻塞点。** 本 README 只讲怎么跑。

> 当前状态一句话：A、B、C 三个检查点都训练完成，但**没有一个技能达到 90%**（A 71–77%、B 20%、C 未测且专家上限只有 68.7%），Foresight / B′ 尚未开始。预期与实测的差距都在 `EXPERIMENTS.md` 里写明了，没有粉饰。

## 环境

- Linux 或原生 Windows 均可跑训练与评估。
- **采集需要 `mplib`**（官方 planner 依赖），它只有 Linux wheel。Windows 下可以训练和评估，但不能运行 `charger.collect`。
- 实测环境：Python 3.12、torch 2.5.1+cu121、mani_skill 3.0.1、sapien 3.0.3、RTX 2080 Ti 22 GiB。
- Windows + WSL 本机实测：WSL Ubuntu-22.04 里 `mplib 0.1.1`、CUDA 均可用，但 **SAPIEN 初始化渲染时崩溃**（`vk::createInstanceUnique: ErrorIncompatibleDriver`，缺 Vulkan ICD），即使 `render_backend='cpu'` 也不行；Windows 端渲染正常但缺 `mplib`。两端各缺一半，采集前需要先解决其中之一。

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows
.venv/bin/pip install -r requirements.txt          # Linux
```

Windows 的 PowerShell 入口 `run.ps1` 只是转发到 `python -m charger.<命令>`，会自动使用 `.venv`，也可用 `CHARGER_PYTHON` 指定解释器：

```powershell
.\run.ps1 test
.\run.ps1 train --data data/formal_rgb --stage A --proprio --relative-actions
```

等效的直接调用：`python -m charger.<module> ...`（需 `PYTHONPATH` 指向仓库根目录）。

## 数据

数据不进仓库（`.gitignore` 已排除 `data/`、`runs/`）。正式数据在 `data/formal_rgb`：1000 个独立 seed 的尝试，**569 条成功示范**，双相机 128×128（`hand_camera`、`base_camera`），按 seed 划分为 train 457 / val 56 / test 56。

重新采集（需 Linux + `mplib`）：

```bash
python -m charger.collect --out data/formal_rgb --seeds 1000 --repeats 1 \
    --obs-mode rgb --render-backend cpu
```

阶段切分（官方 env 真值，仅作标签，不给策略）：

- **A**：`agent.is_grasping(charger)` 首次为真，该动作属于 A。
- **B**：保持抓持，`charger` 到 `goal_pose * Pose([-0.05,0,0])` 的距离 ≤ 1 cm 且姿态误差 ≤ 10°。
- **C**：官方 `env.evaluate()['success']`。

如果想把某一段技能单独切成一个文件（可选，不影响默认训练）：

```bash
python -m charger.split_stages --data data/formal_rgb --out data/split_A --stage A
```

## 训练

```bash
python -m charger.cache_frames --data data/formal_rgb --stage A --camera hand_camera --out data/frame_cache
python -m charger.train --data data/formal_rgb --stage A --camera hand_camera --proprio \
    --relative-actions --frame-cache data/frame_cache --steps 12000 --batch-size 64 --amp --out runs/formal
```

- `charger.cache_frames` 把某个技能的帧导出成可 mmap 的 `.npy`。HDF5 里 RGB 是 gzip 分块，shuffle 后随机读会慢一个数量级，导缓存后接近 GPU 速度。
- `charger.train_forever` 是训练守护进程：子进程崩溃后自动 `--resume last.pt`，只有跑到 `--steps` 才结束。实测 A/B/C 各 12000 步分别约 9 / 11 / 11 分钟。
- 模型：ResNet18（**随机初始化**，无预训练，BN 换 GroupNorm）+ 空间 keypoint 加权回归；条件 = 视觉 + proprio；动作头是 3 层 Transformer（d_model 256）+ 100 步 cosine DDPM，预测 16 步 chunk。总参数 12.95 M。

## 评估

```bash
# 单条链
python -m charger.evaluate --mode A --episodes 100 --budget 200 --render-backend cpu \
    --checkpoints runs/formal --out runs/eval/A_100.json

# 多条链批量跑并出一张表
python -m charger.eval_batch --modes BC BprimeC ABC ABprimeC --episodes 100 --budget 150 \
    --checkpoints runs/formal --out runs/eval/paired
```

评估是**真实 rollout**，不看 loss 猜成功率：

- 每 4 步重新预测一次（16 步 chunk 只执行前 4 步），DDPM 采样 20 步去噪。
- `A` 用全新 seed；`B`/`C`/`BC` 从 held-out 专家前辈回放（恢复真实接触力，不用 `set_state_dict`）；`AB`/`ABC` 链内不 reset。
- 结果写 Wilson 95% 区间。**同一配置重跑会有约 ±3 个点的差异**，原因见 `EXPERIMENTS.md`，正式对比请跑多个 policy seed。

## 数据校验与复现性工具

```bash
python -m charger.validate_shards --data data/formal_rgb --out runs/validation/physics_all.json --shards 4
python -m charger.repro --seed 10001 --out-dir runs/repro/10001
python -m charger.summarize --runs runs --out runs/summary/results.json
```

- `validate_rgb` / `validate_shards`：单次正演逐条校验阶段边界能否复现、episode 尾部是否被污染、最小插入距离。全量 569 条约 6.5 分钟。
- `repro`：逐 replan 记录两次 rollout 的模型输入，定位“从哪一步开始不一样”。
- `summarize`：把数 MB 的评估结果压缩成 `runs/summary/results.json`（进仓库的那份）。

## 诊断视频

```bash
# 单条 rollout，双视角 + 标注（步数 / 距离 / 姿态角 / 是否抓持）
python -m charger.record_video --mode A --seeds 10000 10001 --out runs/videos_A --scale 3
python -m charger.record_video --mode B --seeds 0 205 --data data/formal_rgb --out runs/videos_B

# 成功 vs 失败并排对齐播放
python -m charger.compare_video --mode A --seeds 10000 10001 --out runs/videos_compare_A
```

`record_video` 的 `--mode B` 必须配合 `--data` 使用：B 的起点是 held-out 专家 A 边界，需要先回放专家前缀恢复真实接触。从 plain reset 开始的 B/C 评估无效。

## 问题诊断顺序（实测有效）

发现技能失败时，按这个顺序排查，每一步都能排除一整类原因：

1. **专家能不能复现？** `validate_shards`。A 569/569、B 边界 0 条不符、C 只 391/569。
2. **失败是卡死还是走偏？** `record_video` 看 `grasp` 和距离序列。
3. **是不是评估协议错了？** 检查是否漏了专家前缀（B/C）、是否用了正确的 seed 起点。
4. **换个输入能不能修？** 用对照实验（如 `--camera hand_camera,base_camera`），不靠猜。

## 测试

```bash
python -m unittest discover -s tests -v      # 或 .\run.ps1 test
```

覆盖数据契约（边界动作归属、策略不读真值、seed 隔离、chunk 不跨技能）、DDPM 采样与 residual PPO 语义。

## 目录

```
charger/          采集、训练、评估、校验的全部模块
tests/            单元测试
archive/          已废弃的模块（保留用于复现历史实验结论，不参与主流程）
runs/summary/     进仓库的精简实验结果与数据审计
EXPERIMENTS.md    全部实测数字、失败分析与当前阻塞点
FORESIGHT.md      Foresight Residual RL 的对接约定
STATUS.md         早期试跑记录（历史，已被 EXPERIMENTS.md 取代）
```

## 引用资料

- [PlugCharger-v1 任务页](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#plugcharger-v1)
- [ManiSkill 官方采集](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html)
- [官方 plug_charger 规划器](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/solutions/plug_charger.py)
- [Foresight Residual RL (arXiv:2607.16506)](https://arxiv.org/abs/2607.16506)
