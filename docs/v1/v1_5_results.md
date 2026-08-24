# V1.5 结果汇总

> 本文档合并 V1.5.1、V1.5.2 与 V1.5.3 的实验结果、核对结果和人工评审结论，作为 V1.5 阶段的唯一结果入口。

## 1. 阶段结论

V1.5 已确认：`A2_highres_only` 相对 residual-matched A0 具有独立的路径风格增益；高分辨率参考注入同时提高局部几何风险。现有 masking 与 SVD purification 在固定配置、多配对、多 seed 下没有形成稳定的几何安全优势，本阶段结束该路线。

## 2. V1.5.1：因果诊断与核对

- 完成 A2 与 residual-matched A0 的路径对照；
- 确认 A2 的风格增益具有路径层面的独立来源；
- 记录并核对 timestep、seed、配对和输出路径；
- 几何风险主要表现为建筑形态放大、错误硬边、屋顶/塔尖/立面变形、背景结构建筑化，以及语义对应但几何不安全的迁移。

## 3. V1.5.2：Purification 结果

reference-only purification、masking 和 SVD purification 在多配对、多 seed 条件下未形成稳定的几何安全优势。它们改变了参考 token 的统计特征，但仍不足以阻止局部几何被高分辨率路径接管。

## 4. V1.5.3：冻结多 seed 结果与人工评审

V1.5.3 冻结了多 seed 评估与人工评审结论：A2 的风格优势可以重复观察，但几何失败也会随高分辨率参考注入重复出现；purification 变体没有显示足够稳定、可推广的改善。本阶段停止继续堆叠 reference-only purification 变体。

## 5. 方法与样本约束

- 正式案例名统一使用 `v1_5_kulhanek_snow_winter`；
- 旧 `v1_5_kulhanek_snow_street` 仅作为历史 alias；
- 结果解释必须同时考虑配对、seed、内容刚性和局部失败类型；
- V1.5 完成风格增益与 purification 对照，几何风险留给 V2.0 继续验证。

## 6. 阶段状态

```text
V1.5.1 causal diagnostics: CLOSED
V1.5.2 purification sweep: CLOSED
V1.5.3 frozen multiseed review: CLOSED
V2.0 geometry-risk validation: CLOSED
```

## 7. 后续方向

V2.0 将验证内容侧几何风险图是否能够预测 A2 真实局部几何失败；在验证完成前，不实现空间 residual gate、Geometry-Gated HighRes Boost、pair compatibility 或 learned router。
