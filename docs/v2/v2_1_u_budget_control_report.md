# V2.1 `U_budget` 控制报告

## 状态：已关闭

`U_budget` 用来区分两个因素：总 reference residual 预算和区域分配。它将 Subject-oriented 结果与尽量匹配的全局 uniform A2 预算进行比较。

## 主要结果

匹配总预算后，`S_raw` 相比 `U_budget` 只有有限的 Subject 风格集中优势；Background 保护基本相同，也没有清晰的额外几何安全收益。视觉改善主要体现为轻微颜色/风格集中，而非可靠的结构保护。

Snow 的 coarse token overlap 还会让标量增益放大混合 token，导致 `S_match` 出现结构/风格接管。一次 budget match 未形成统一的空间路由优势。

## 决策

V2.1 区域分配路线关闭。保留 `S_raw`、`S_sep_neutral` 等结果作为消融和诊断对照，主线固定 A2 路径并进入 V2.2a 全局参考强度前沿。
