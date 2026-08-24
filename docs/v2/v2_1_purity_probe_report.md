# V2.1 纯度感知 Snow 探针

## 状态：已关闭

这是 V2.1 的诊断性归档，不建立新的空间路由方法，也不扩展到更多 seed。

## 范围

本轮没有新增 Church/Wave 结果，只生成两个 Snow seed42 结果：`S_sep_neutral` 与 `S_sep_conservative`，纯度阈值为 `0.80`。

## 平均池化纯度审计

Snow 的 16×16 token 分类为：纯 Subject `0`、Mixed `40`、纯 Background `152`、无效 `64`。这与此前的 max-pool 诊断不同：一个被 Subject 触碰的 token 不会自动被视为完整 Subject token。像素级 Subject/Background 无重叠；无效 token 的 route gain 为零；Mixed token 不会超过配置的 mixed gain。

## 路由定义和自动结果

当前 S 的增益解释为 Subject `1.0`、Background `0.0`：`S_sep_neutral` 对纯 Subject/Mixed 使用 `1.0`，纯 Background 使用 `0.0`；`S_sep_conservative` 只对纯 Subject 使用 `1.0`。

相对于同分辨率 `U` 的 residual 比例：

| 变体 | 64×64 | 32×32 | 16×16 |
|---|---:|---:|---:|
| `S_raw` | 0.132 | 0.180 | 0.260 |
| `S_match` | 0.467 | 0.744 | 1.330 |
| `S_sep_neutral` | 0.260 | 0.283 | 0.389 |
| `S_sep_conservative` | 0.042 | 0.029 | 0.000 |

## Snow seed42 人工复核

`S_raw`、`S_sep_neutral` 和 `S_sep_conservative` 均没有出现明显的山体建筑化；`S_match` 出现中心建筑融合并被拒绝。三个 Subject-oriented 结果的视觉差异很小，纯度路由只带来轻微饱和度下降，Background 保护仍然有效，也没有明显 mask 边界伪影。

因此保留 `S_sep_neutral` 作为较少抑制的临时候选；阶段停止增加 seed 和扩展空间路由。后续测试采用 `global lambda × S_sep`，Subject-only gain 不再单独增加。

证据目录：`runs/ip_adapter_plus_injection/v2_1_purity_audit/`、`runs/ip_adapter_plus_injection/v2_1_purity_probe/audits/` 和 `runs/ip_adapter_plus_injection/v2_1_purity_probe/v1_5_kulhanek_snow_winter/seed42/`。
