# V2.3 内容—参考图配对响应画像扩展

> 状态：V2.3 已关闭。seed42 筛查和代表性配对的多 seed 验证均已完成。

## 目标

检验不同内容—参考图配对是否具有可区分的 reference-pressure 响应/风险画像。阶段标签采用“初始 takeover 易感性、增量 pressure sensitivity、风格响应”三个维度。

配对设计包含：同一内容换参考图、同一参考图换内容，以及建筑、自然景观、植被、水面、海岸和城市/道路等不同语义与几何关系，从而区分内容效应、参考效应和配对交互。

## 冻结协议

使用固定的 A2 high-resolution-only 调度和统一全局 multiplier：`lambda = 0.2 / 0.4 / 0.6 / 0.8 / 1.0`，`seed = 42`。不使用 rigid、Subject/Background、purity 或其他空间 gate。

自动记录 IP residual、内容图 RGB 变化、参考颜色/统计代理、Canny edge F1、chamfer、相邻强度变化、响应 slope 和归一化 AUC。

## 人工复核

人工表固定为以下字段：

```text
case, seed, lambda,
human_style_score_0_4,
baseline_takeover_0_3,
incremental_takeover_0_3,
style_valid, reference, review_note
```

`lambda=.2` 记录相对 Content 的 `baseline_takeover_0_3`；其余强度记录相对前一档的 `incremental_takeover_0_3`。无效的目标风格使用 `style_valid=false` 和 `human_style_score_0_4=NA`。Style 数值采用 `0–4`，Takeover 采用 `0–3`。

标注表位于 `runs/ip_adapter_plus_injection/v2_3_pair_response_profiles/audits/human_sensitivity_annotations.csv`。

## seed42 结果与多 seed 复核

首轮清单为 `configs/experiment/v2_3_pair_response_profiles.csv`，排除了三个 canonical pairs。结果位于 `runs/ip_adapter_plus_injection/v2_3_pair_response_profiles/`。

seed42 筛查形成 10 个新配对的完整五档响应曲线。以下 6 个代表性配对已完成 seed123/777 复核：

| 配对 | 代表画像 |
|---|---|
| `compat_G1_church` | 高风格响应、低风险 |
| `clean_demuth_G1_water_lake` | 同一 Demuth 参考下的高质量迁移 |
| `clean_demuth_G1_forest` | 同一 Demuth 参考下的低响应 |
| `compat_G2_opposite_wave` | 低强度初始风险、后期继续增加 |
| `compat_G4_sea_cliff_wave` | 高初始风险、后续趋于饱和 |
| `clean_demuth_G4_city_mismatch` | 最低测试强度即出现严重 takeover |

6 个代表配对已补充 `seed123/777` 的完整 `lambda=.2/.4/.6/.8/1.0` 曲线。复核结果保留配对级响应差异，同时记录 seed 对局部变化的影响。V2.3 阶段状态为 **CLOSED**。

下一阶段进入 V2.4 生成前 Pair Preflight 分析，先整理 content/reference 特征与已有响应画像，再决定是否形成简单规则或轻量 controller。
