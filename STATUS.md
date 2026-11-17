# 状态摘要

此文件已被 [`EXPERIMENTS.md`](EXPERIMENTS.md) 取代，仅保留为历史索引。

## 当前状态（2026-11-17）

| 目标 | 实测 | 状态 |
|---|---|---|
| 官方 collect 数据，训练只用 wrist | 569 条成功示范，训练用 `hand_camera` | 完成 |
| A/B/C 各自 SR ≥ 90% | A 71–77%、B 20%、C 未测（专家上限 68.7%） | 未达标 |
| Foresight RL 微调 B → B′ | 代码有骨架，实验未运行 | 未开始，前置条件不满足 |

完整数字、失败原因、已证伪的中间结论见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。
早期（2026-11-16 及以前）的试跑记录见 [`docs/HANDOFF_2026-11-16.md`](docs/HANDOFF_2026-11-16.md)，其中关于“渲染改变物理”和“暂停 C 的 RGB 采集”的结论已被实测推翻。
