# 样本数据：3 个 seed × 3 个 skill

来源 `data/formal_rgb`（官方 motion-planning 示范），seeds [1, 2, 3]。
每一步都包含两个相机：`hand_camera`（wrist）和 `base_camera`（全局）。

| skill | seed | 步数 | success 在段内为 True 的步数 |
|---|---|---|---|
| A | 1 | 78 | 0 |
| A | 2 | 82 | 0 |
| A | 3 | 77 | 0 |
| B | 1 | 34 | 0 |
| B | 2 | 36 | 0 |
| B | 3 | 28 | 0 |
| C | 1 | 35 | 1 |
| C | 2 | 37 | 1 |
| C | 3 | 36 | 1 |

## 结构

```
<skill>/trajectory.h5
  traj_seed<seed>/
    actions          (T, 8)            专家关节目标 + 夹爪
    success          (T,)   bool       该步是否满足官方 success（C）
    terminated       (T,)   bool
    truncated        (T,)   bool
    obs/sensor_data/{hand_camera,base_camera}/rgb   (T,128,128,3) uint8
    obs/sensor_param/{cam}/intrinsic_cv, extrinsic_cv, cam2world_gl
    obs/agent/qpos   (T, 9)   qvel (T, 9)
    obs/extra/tcp_pose (T, 7)
<skill>/index.json    每个样本的 seed、原始 trajectory、起止步号、reset_kwargs
```

`actions` 与图像已按该 skill 的边界切片（A 抓取 / B 对齐 / C 插入）。
原库里整条 episode 是连续的，这里切出来是为了单独看每个 skill。
