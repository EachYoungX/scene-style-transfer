# V2.4 生成前 Pair Preflight 分析

> 状态：进行中。第一轮完成生成前特征提取和 pair profile 对照，暂不生成新图。

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

本地当前未配置 CLIP 或 DINOv2 权重。粗尺度 patch 项使用 RGB/灰度统计作为可重复的基础特征；CLIP global cosine、DINOv2 global similarity 和 DINO patch similarity 列为后续扩展项。

## 4. 第一轮结果

`analysis/v2_4_feature_correlations.csv` 使用 Spearman 检查生成前特征与 baseline susceptibility、style gain、incremental risk 的关系。当前样本量用于相关性筛查，classifier 放到样本扩充后。

已完成 13 个 pair 的生成前特征提取和 profile 聚合，其中 10 个 pair 的标签来自 V2.3，3 个 V2.2 canonical pair 的标签已从 targeted multiseed 人工表转换到 V2.4 schema。整个过程没有生成新图。

当前筛查中较值得继续验证的方向是：

- 初始风险：Content Canny density 与 baseline takeover 的 Spearman 绝对值最高；
- 增量风险：reference/content LSD line-density ratio 与 `incremental_takeover_max_median` 的相关性最高；
- 风格响应：当前 12 个有效 Style pair 上的几何长度和方向差异只有探索性相关，不能作为单独的风格兼容性判据。

这些数值只是 13 个已评分 pair 上的探索性排序，不是泛化性能。`analysis/v2_4_preflight_checks.csv` 另外记录了初始风险分组、Demuth controlled subset 和低风险风格响应分组；最后一组仍然样本严重不平衡。

后续扩展 pair 时采用 content/reference 留出检查，测量 pair compatibility 的可迁移性。

## 5. 当前行动项

1. 复核 3 个 canonical pair 的新 profile 是否与人工总体判断一致；
2. 在新增 pair 前，复核 Demuth reference 的 controlled subset 排序是否具有语义解释；
3. 新增 10–20 个 pair 验证初始风险与增量风险候选特征；
4. 保持 content/reference 留出检查，不进行 random pair split；
5. 特征规律稳定后再设计 `Reject / 0.2 / 0.6 / 1.0` 的简单 preflight 规则。
