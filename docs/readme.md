# 项目文档索引

## 当前阶段

项目当前处于 **V2.4 生成前 Pair Preflight 分析阶段**。V1、V1.5、V2.0、V2.1、V2.2a 和 V2.3 已完成并关闭。V2.4 使用已有配对结果和生成前图像特征，检查不同 content-reference pair 的响应画像是否具有可解释差异。

## 主要入口

| 文档 | 用途 |
|---|---|
| [V1 阶段结果汇总](v1/v1_0_results.md) | 从基线到 V1.5 的完整实验链、结论和未解决问题 |
| [V1.5 结果汇总](v1/v1_5_results.md) | V1.5.1–V1.5.3 的冻结结果 |
| [V2.0 标注执行指南](v2/v2_0_annotation_evaluation_protocol.md) | 几何风险标注格式与人工标注规则 |
| [V2.0 局部刚性门控审计](v2/v2_0_local_rigid_gate_audit_report.md) | 局部 residual gate 的实现审计与负结果 |
| [V2.1 区域风格试验](v2/v2_1_regional_style_pilot.md) | Subject/Background 区域分配试验 |
| [V2.2a 强度前沿](v2/v2_2a_global_safe_strength_frontier.md) | 全局参考压力与响应画像的验证 |
| [V2.3 执行方案](v2/v2_3_pair_response_profile_expansion.md) | 配对响应画像的固定协议与后续执行 |
| [V2.3 阶段总结](v2/v2_3_pair_response_profile_summary.md) | seed42 人工评估、画像分类和阶段决定 |
| [V2.3 City-Mismatch 审计](v2/v2_3_city_mismatch_audit.md) | 城市失配配对的颜色与参考泄露核查 |
| [V2.4 生成前 Pair Preflight 分析](v2/v2_4_pair_preflight_analysis.md) | 生成前配对特征与响应画像分析 |

## 阅读顺序

建议按以下顺序阅读：

1. [V1 阶段结果汇总](v1/v1_0_results.md)，了解问题发现和方法收缩过程；
2. [V1.5 结果汇总](v1/v1_5_results.md)，了解路径与 purification 对照；
3. 当前 V2.x 报告，了解已完成实验和正在执行的分析。

项目计划、图像数据库说明和历史研究笔记位于本地 `internal_docs/`，不纳入公开文档索引。
