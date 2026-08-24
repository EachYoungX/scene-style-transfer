# V2.1 区域分配审计

## 状态

当前 `S_subject` / `S_background` pilot 已完成审计并关闭。预算匹配问题已由 `U_budget` 对照暴露并记录，但证据不足以把空间分配提升为主方法，也没有冻结统一的 `S*`。

审计使用三个案例、seed `42/123/777` 的现有 V2.1 residual 日志，并用 seed42 做一次只记录区域 residual 的审计 forward；审计 forward 不保存生成图像。

## 原始总注入预算

总比例是相对于 `U` 的 gated residual energy 平方根，汇总 `64/32/16` 三个分辨率：

| 案例 | `S_subject / U` | `S_background / U` |
|---|---:|---:|
| Church | 0.266 | 0.766 |
| Snow | 0.143 | 0.784 |
| Wave | 0.393 | 0.601 |

`S_subject` 的预算明显低于 `U`，尤其是 Snow；原始 `S_background` 更接近 `U`，与其视觉结果接近 `U` 一致。

## 16×16 区域审计

下表是 `gated / raw` 的能量加权 RMS 比例：

| 案例 | 变体 | 全局 | Subject | Background |
|---|---|---:|---:|---:|
| Church | `S_subject` | 0.451 | 1.000 | 0.459 |
| Church | `S_background` | 0.910 | 0.900 | 1.000 |
| Snow | `S_subject` | 0.275 | 1.000 | 0.313 |
| Snow | `S_background` | 0.892 | 1.000 | 1.000 |
| Wave | `S_subject` | 0.505 | 1.000 | 0.588 |
| Wave | `S_background` | 0.745 | 0.838 | 1.000 |

Subject 自身的 active token 没有被稀释，但粗分辨率 mixed cell 中的 Background token 仍会获得 residual。`S_background` 在 16×16 仍会保留大量 Subject 区域 residual，因此视觉上可能接近 `U`。

## 16×16 token 覆盖

| 案例 | Subject active | Background active | 重叠 | 重叠/Subject |
|---|---:|---:|---:|---:|
| Church | 31 | 185 | 24 | 77.4% |
| Snow | 20 | 192 | 20 | 100.0% |
| Wave | 61 | 110 | 43 | 70.5% |

Snow 中每个 active Subject token 都同时属于 Background gate，这是 coarse-support 行为的直接机制。

## 预算匹配与因果探针

首轮 `S_match` 使用 `subject_gain=3.5`，seed42 总比例为 Church `0.989`、Snow `0.538`、Wave `1.454`，不作为统一全局 `S*`。该设置会放大 mixed token，并引入结构/风格接管。

当前实现中的 `S0`（Background=0）与既有 `S_subject` 完全相同；`B0`（Subject=0）与既有 `S_background` 完全相同，已有图像可直接复用。

## 最终解释

`S_raw` 只有轻微风格优势，Background 保护几乎相同，未显示额外几何安全收益。V2.1 关闭，转入 **V2.2a 全局安全强度前沿**：固定 A2 路径，只改变统一全局 reference multiplier。

证据位于 `runs/ip_adapter_plus_injection/v2_1_regional_pilot/audits/v2_1_allocation/`、`runs/ip_adapter_plus_injection/v2_1_allocation_audit/audits/summary/` 和 `runs/ip_adapter_plus_injection/v2_1_smatch_pilot/`。
