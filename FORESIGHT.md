# Foresight Residual RL 对接约定

论文：[Foresight Residual RL for Long-Horizon Robot Manipulation with Vision-Language-Action Models](https://arxiv.org/abs/2607.16506)，v1，2026-07-17。

## 论文方法

依据 Algorithm 1–2、Section V 和 VI：

1. 冻结 base policy，训练 residual policy；执行动作是 `a_base + alpha * a_residual`，不是直接更新 base 权重。
2. 在 base policy 的成功终止状态采集图像；从每个状态多次执行固定的下游策略，用成功次数 / 总次数作为 foresight 标签。
3. 用 binomial NLL 训练视觉概率预测器。
4. 使用 PPO 训练 residual，终止奖励为 `当前技能成功指示 * 预测的下游成功概率`；论文此处没有额外 dense shaping。
5. 完整方法从最后一个技能开始逆序训练；训练某个技能时，下游策略和 foresight predictor 固定。预测器只用于训练奖励，不参与测试时的动作或切换决策。

论文具体实现采用 π₀、DINOv2 和 Isaac Gym；本项目使用从头训练的 ResNet18、视觉动作扩散策略和 ManiSkill，属于方法迁移，不是原论文逐项复现。

## 本项目当前 validation 的定义

保持用户原始范围：先学习 A/B/C，然后只改进 B，固定 A 和 C。

- `B′ = 冻结 B + residual_B`，以 C 的成功概率构造 B 的 foresight 奖励。
- `BC` 与 `B′C` 使用相同的 held-out 专家 A 前缀、相同环境 seed 和配对的随机策略 seed。
- `ABC` 与 `AB′C` 使用相同的新 reset seeds；A 为同一检查点，链内不 reset。
- 需要额外记录 A→B 接口分布上的 `AB` / `AB′` 阶段完成率，以及 B→C 的交接质量，才能定位退化发生在哪里。仅全链下降不能单独证明 AB 接口本身退化。
- `SR(B′C) > SR(BC)` 且 `SR(AB′C) < SR(ABC)` 是用户提出的待检验假设，不能作为论文已证明的结论或训练保证。论文完整逆序方法的目标是提高全链性能。
- 可另设仅局部成功奖励的 residual 对照，用来区分普通 residual RL 与 foresight 的影响；不能把这个对照命名为论文完整方法。

## 实现前需要满足的条件

当前 ABC 仅有 2 条示范和各 2 步训练的接口检查点，不能据此生成有意义的 foresight 成功概率或做正式 PPO 实验。先完成 base 技能训练及独立、串联评估。

后续实现时，必须保持动作坐标与缩放一致，保存 residual scale；严格冻结 base 参数和归一化统计，先校验零 residual 能复现 base rollout。复用同一 base action buffer 和 replanning 规则，技能切换时清空 buffer。

重复执行 C 来标注同一个 B 终态时，不能只恢复 `env_states` 并假定抓持接触力相同。应记录并精确重放到 B 边界的 reset、已执行动作序列和策略随机种子，重新建立接触后再启动独立的 C rollout；无效重放必须单独记录。

## 当前实现状态

后续已加入 residual PPO、foresight 数据生成、预测器及配对比较的代码，但尚未进行正式 B′ 训练，也未验证其效果。最新状态与实现差异见 HANDOFF_DEEPSEEK.md。
