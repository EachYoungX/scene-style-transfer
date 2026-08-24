# V2.2a 全局安全强度前沿

> 状态：**已关闭**。试验：三个 canonical pairs，seed `42`，随后扩展到 seed `42/123/777`。

## 研究问题

不同内容—参考图配对是否需要不同的全局 reference injection strength，才能得到更好的风格—结构折中？本阶段先测量安全强度前沿，再决定是否值得设计 learned predictor 或 pair-aware router。

## 固定方法

固定 A2 high-resolution-only 调度，只改变 uniform multiplier：

```text
ip_delta_final = lambda * ip_delta_A2
lambda = .2 / .4 / .6 / .8 / 1.0
```

不使用 rigid mask、Subject/Background mask、purity routing 或其他空间纯化。canonical cases 为 Demuth × church、Kulhanek × snow_winter、Demuth × wave。`U = lambda 1.0` 直接复用既有结果。

## 自动测量

记录全局和 `64×64/32×32/16×16` residual、输出相对内容图的 RGB 差异、各分析区域差异、Canny edge F1、chamfer 以及参考颜色/统计代理。RGB 接近参考图作为响应代理，Style、结构、几何接管和参考泄漏由人工复核。

## seed42 初始结果

实测 residual 比例基本等于请求的 lambda，说明扫描改变的是预期的全局 residual budget，各分辨率保持一致缩放：

| 配对 | λ=.2 | λ=.4 | λ=.6 | λ=.8 | U λ=1 |
|---|---:|---:|---:|---:|---:|
| Church | .200/.200 | .400/.400 | .601/.600 | .800/.800 | 1.000/1.000 |
| Snow | .198/.200 | .398/.400 | .598/.600 | .799/.800 | 1.000/1.000 |
| Wave | .199/.200 | .399/.400 | .599/.600 | .799/.800 | 1.000/1.000 |

每格为 `全局 / 16×16`。输出相对内容图的 RGB MAE 随 lambda 单调增加：Church `21.80 → 37.45`、Snow `25.57 → 28.59`、Wave `24.62 → 33.87`。这些指标作为响应代理，Style 分数沿用人工评估。

## seed42 的初步人工假设

| 配对 | 暂定 λ_safe | 首次可见风险区间 | 更严重的接管 |
|---|---|---|---|
| Church | `.4` | `.4 → .6` | `.8` 开始明显，`1.0` 出现严重建筑语义错误 |
| Snow | `.4` | `.4 → .6` | 没有观察到独立的第二次断点 |
| Wave | `≥1.0` | 未观察到 | 未观察到，测试上限右删失 |

这里的 `λ_safe` 仅记录当前 seed 和离散测试等级中的最高安全档位，不作为稳定的 pair 标签。

## 多 seed 修正

对三个配对、三个 seed、五个 lambda 完成 `3 × 3 × 5 = 45` 条人工复核记录。跨 seed 未形成稳定的 per-pair sharp safe knee：Church 的 `.4→.6` 与 `.8→1.0` 加速受 seed 影响；Snow 在额外 seed 中更平滑；Wave 也总体平滑，只有 seed777 的 `.8→1.0` 更明显。

可靠结论是：

> 增大 lambda 会稳定地产生风格—结构权衡，但风险突然增加的强度点并不跨 seed 对齐。

因此研究目标从 `pair → lambda_safe` 改为 `pair → response profile`，描述风险敏感度、风格敏感度、风格—风险效率，以及平滑或局部加速的响应形状。相邻变化只在同一个 seed 内比较，不比较不同 seed 的绝对泄漏量。

## slope/AUC 分析

主摘要使用 lambda `.2–1.0` 的线性 slope 和归一化梯形 AUC；相邻变化仍用于诊断局部跳变。当前三 seed 结果支持一个较窄的判断：residual 增量几乎相同，但输出响应敏感度不同，Church 的 RGB 内容变化 slope 约为 Snow 的四倍、Wave 的两倍。

RGB 响应敏感度与几何风险敏感度分开记录，人工 takeover/style 分数和相关性检查用于建立两者关系。颜色/对比度字段作为参考响应代理。

## V2.2 决策

**V2.2 canonical response–risk validation 已关闭。** 全局参考压力产生配对依赖的输出响应；per-pair sharp risk threshold 在多 seed 中未形成稳定对齐；自动几何风险代理仍需人工评分校准。下一阶段扩展配对多样性。

## 证据位置

结果目录为 `runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/`；运行脚本为 `scripts/run_v2_2a_safe_strength_frontier.py`，评价和 review 脚本为 `scripts/evaluate_v2_2a_safe_strength_frontier.py` 与 `scripts/make_v2_2a_frontier_review_grid.py`。
