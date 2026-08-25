# V2.4 生成前 Pair Preflight 分析

> 状态：V2.4 pair-preflight branch 已关闭。V2.4a hypothesis、V2.4b 23-pair expansion、V2.4c validation、V2.4d feasibility probe 和最终 global λ freeze 均已完成；pair-aware controller 不进入正式方法。

## 1. 研究目标

V2.4 研究 Content—Reference pair 是否可以在生成前被归入可解释的响应画像：

- 初始 takeover 易感性；
- reference pressure 增大后的风险敏感度；
- 有效 Style 响应能力。

本阶段使用已有 13 个 pair。V2.3 的代表性多 seed 评分作为已完成 pair 的标签来源；V2.2 canonical pair 的人工数值表处于待补状态，profile 表保留对应记录并标记状态。

## 2. Pair profile 字段

`analysis/v2_4_pair_profiles.csv` 记录：

```text
baseline_takeover_median
baseline_takeover_max
style_at_02_median
style_at_10_median
style_gain_median
incremental_takeover_max_median
incremental_nonzero_interval_count
late_escalation_frequency
style_valid_rate
seed_count
profile_label
```

`lambda=.2` 的 baseline takeover 与后续 λ 的 incremental takeover 分开统计。ordinal 分数只用于排序和分组，不做线性倍数解释。

## 3. 第一轮生成前特征

`analysis/v2_4_pair_features.csv` 使用与生成一致的 `512×512 fit_square_crop` 输入，提取：

- Content/Reference Canny edge density；
- LSD line density、长度统计和方向分布；
- edge density ratio、line density difference、方向熵差异；
- Lab 均值/标准差距离、颜色直方图距离、对比度差异；
- Laplacian 与高频能量差异；
- 16×16 粗尺度 RGB patch 最近邻相似度和 mutual-nearest fraction。

粗尺度 patch 项使用 RGB/灰度统计作为可重复的基础特征；representation 扩展使用本地 IP-Adapter CLIP vision checkpoint 与 `facebook/dinov2-small`。

## 4. 第一轮结果

`analysis/v2_4_feature_correlations.csv` 使用 Spearman 检查生成前特征与 baseline susceptibility、style gain、incremental risk 的关系。当前样本量用于相关性筛查，classifier 放到样本扩充后。

已完成 13 个 pair 的生成前特征提取和 profile 聚合，其中 10 个 pair 的标签来自 V2.3，3 个 V2.2 canonical pair 的标签已从 targeted multiseed 人工表转换到 V2.4 schema。整个过程没有生成新图。

当前筛查中较值得继续验证的方向是：

- 初始风险：Content Canny density 与 baseline takeover 的 Spearman 绝对值最高；
- 增量风险：reference/content LSD line-density ratio 与 `incremental_takeover_max_median` 的相关性最高；
- 风格响应：当前 12 个有效 Style pair 上的几何长度和方向差异只有探索性相关，不能作为单独的风格兼容性判据。

这些数值只是 13 个已评分 pair 上的探索性排序，不是泛化性能。`analysis/v2_4_preflight_checks.csv` 另外记录了初始风险分组、Demuth controlled subset 和低风险风格响应分组；最后一组仍然样本严重不平衡。

## 5. V2.4b representation features

已在 `sst_env` 中使用本地 IP-Adapter CLIP vision checkpoint 和 `facebook/dinov2-small` 完成 13 个 pair 的 generation-free representation extraction。结果位于：

- `analysis/v2_4_pair_feature_analysis.csv`：每行一个 pair，包含 profile、低级几何、RGB patch、CLIP 和 DINOv2 特征；
- `analysis/v2_4_pair_feature_correlations.csv`：五个 target 的 Spearman 筛查；
- `analysis/v2_4_feature_effects.csv`：`baseline=0` 对 `baseline>=2` 的组间 median 与 Cliff's delta；
- `analysis/v2_4_demuth_controlled_subset.csv`：Demuth fixed-reference subset 的逐 pair 对齐表；
- `analysis/v2_4b_feature_manifest.json`：模型、设备和特征清单。

当前 13 pair 上，Content Canny density 对初始风险的方向最稳定；reference/content LSD ratio 对增量风险更突出；CLIP patch mutual correspondence 和 DINO patch similarity 对 Style gain 出现探索性信号。它们仍然不能直接用于 controller 训练。

后续扩展 pair 时采用 content/reference 留出检查，测量 pair compatibility 的可迁移性。

## 6. V2.4b targeted profile balancing

已完成 10 个 generation-free preflight 后的 controlled-cross 候选，配置位于 `configs/experiment/v2_4b_targeted_profile_candidates.csv`。每个候选使用 seed42、λ=.2/.4/.6/.8/1.0、无 mask，共 50 张输出。审阅材料位于：

- `runs/ip_adapter_plus_injection/v2_4b_targeted_profile_candidates/reviews/all_cases_targeted_seed42.png`；
- `runs/ip_adapter_plus_injection/v2_4b_targeted_profile_candidates/reviews/cases/`；
- `runs/ip_adapter_plus_injection/v2_4b_targeted_profile_candidates/audits/human_sensitivity_annotations.csv`。

目标是补齐 P1/P2，而不是继续扩大 P4。

## 7. 当前行动项

1. V2.4 pair-preflight branch 保持关闭；
2. 使用 `configs/experiment/v2_4_final_fixed_a2.yaml` 的固定 A2 operating point；
3. 进入 external same-protocol benchmark，不再开展 pair-aware controller 探索。

## 8. V2.4b seed42 人工结果

10 个 pair 的初步 profile 分布为：6 个 `P1_low_risk_high_response`，1 个有效的 `P2_low_response`，1 个无效风格响应的 `lake_monet` 控制 pair，以及 2 个整组 `style_valid=FALSE` 的 forest pair。所有 pair 的 λ=.2 baseline takeover 均为 0；增量 takeover 只在 `church_kulhanek`、`lake_hokusai`、`wave_monet` 出现，且均为低等级、有限区间信号。该结果用于代表性 multiseed 选择，仍属于 seed42 screening，不作为 controller 训练标签。

## 9. V2.4c common-seed validation

23 个 pair 已统一使用 seed42 标签和同一批 42 个 generation-free features，结果位于：

- `analysis/v2_4c_common_seed_profiles.csv`；
- `analysis/v2_4c_common_seed_correlations.csv`；
- `analysis/v2_4c_common_seed_effects.csv`；
- `analysis/v2_4c_common_seed_manifest.json`。

`style_valid` 已作为独立 target。23 个 pair 中 20 个为有效风格迁移，3 个为无效：两个 Forest controlled pair，以及已有的 `compat_G4_city_mismatch`。无效 pair 的 `style_gain_if_valid` 保持 NA，不参与 conditional style responsiveness 相关性。

Common-seed screening 继续支持三条方向：`content_canny_density` 与 initial susceptibility，CLIP patch mutual correspondence / DINO patch similarity 与 valid-pair style gain，reference/content LSD length ratio 与 pressure escalation。已有 3-seed 子集的方向确认位于：

- `analysis/v2_4c_multiseed_profiles.csv`；
- `analysis/v2_4c_multiseed_correlations.csv`；
- `analysis/v2_4c_direction_comparison.csv`。

这些结果用于方向确认，不把 23 个 seed42 profile 与 3-seed median 混合成单一训练表。多 seed 子集没有新的 `style_valid=FALSE` pair，因此 viability target 仍需后续专门验证。

## 10. V2.4d minimal feasibility probe

使用 1–3 个冻结特征、Logistic probe，以及 leave-one-content-family-out / leave-one-reference-family-out：

- `analysis/v2_4d_feasibility_probe.csv`；
- `analysis/v2_4d_feasibility_summary.csv`。

结果仍是 exploratory only。style viability 和 conditional style responsiveness 的 family-holdout 表现接近 chance；pressure 的个别 split 可达到较高 balanced accuracy，但跨 group median 仍不稳定。当前证据支持继续收集受控 multiseed 复核，不支持直接训练正式 controller。

## 11. V2.4e global operating-point freeze

对 23 个 pair 的 seed42 common-screening 表统一比较 λ=.2/.4/.6/.8/1.0，结果位于：

- `analysis/v2_4e_global_lambda_summary.csv`；
- `analysis/v2_4e_global_lambda_decision.json`；
- `configs/experiment/v2_4_final_fixed_a2.yaml`。

冻结规则为：先排除 active takeover rate >25% 或 severe takeover rate >20% 的候选，再最大化 `style_valid AND style_score>=2` 的有效风格覆盖率，最后选择最小 λ。结果选择 `λ=0.6`：有效风格覆盖率为 56.5%，active takeover rate 为 21.7%，severe takeover rate 为 0%；λ=.8 和 1.0 的 active takeover rate 分别升至 30.4% 和 43.5%。

因此最终内部 operating point 为：

```yaml
method: A2_fixed
schedule: A2_highres_only
reference_strength: 0.6
case_adaptive_lambda: false
```

V2.4 的 generation-free features 保留为机制分析和 discussion evidence。`Pair-aware preflight prediction was not sufficiently robust under family-held-out evaluation.` 正式 benchmark 从固定 A2 operating point 进入外部同协议比较。
