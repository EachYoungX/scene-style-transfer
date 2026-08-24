# V2.1 区域风格试验

> 状态：已关闭。首轮试验：`U / S_subject / S_background`，seed `42`。

## 最终决定

区域级分配可以把有限参考影响集中到 Subject，但在匹配总体注入强度后，相比 uniform attenuation 只有轻微风格优势，没有清晰的额外几何安全收益。`S_raw` 仅作为消融/控制保留，不提升为主方法，也不继续做 Subject/Background gain、purity threshold、Mixed routing、`R+S` 或完整多 seed 扫描。

下一阶段转向更直接支持的假设：不同内容—参考图配对可能需要不同的全局参考强度。

## V2.0 承接

V2.0 局部 rigid gate 审计已关闭。实现本身正确，但 IP-Adapter residual 级门控没有可靠锁定最终几何：Snow 记录为区域/粗粒度支撑失败候选，Church 记录为边缘局部失败候选。两者保留为诊断分类。

## 首轮范围

固定 A2 high-resolution-only 调度，只比较 `U`（uniform A2 image-branch 注入）、`S_subject`（仅注入有效 Subject 区域）和 `S_background`（仅注入有效 Background 区域）。形式案例是 Church、Snow、Wave，seed `42`，并为三者加入 `U_budget` 控制。没有加入 Neutral 注入、halo 扫描、learned router、timestep 搜索或几何失败重标注。

## Mask 规则

沿用 `data/derived/v2_0_geometry_risk/annotations/soft_stylization/` 下的 `_S` 和 `_B` 文件。加载器将 RGB 转灰度并以 `128` 阈值化，输出严格二值 PNG。有效区域排除 `valid_eval`、rigid 区域和黑色 padding；Subject 与 Background 重叠时使用 Subject 优先。没有手绘 Neutral，Neutral 只表示被排除或未分配的像素。

运行：

```bash
python scripts/prepare_v2_1_region_masks.py
python scripts/run_v2_1_regional_pilot.py --overwrite
```

区域 gate 使用 adaptive maximum pooling：token 的源像素只要有一个属于目标区域就激活。这与 V2.0 为保留细线而使用 minimum pooling 的 rigid retain gate 不同。

## seed42 自动结果

`U` 与已有 V2.0 uniform A2 输出逐像素一致。有效像素统计如下：

| 案例 | Subject | Background | derived Neutral | 刚性排除的 Subject |
|---|---:|---:|---:|---:|
| Church | 16,459 | 149,571 | 0 | 4,150 |
| Snow | 1,935 | 162,446 | 0 | 4,783 |
| Wave | 37,280 | 89,212 | 0 | 0 |

区域差异并不等同于结构改善；最终结论以 `U_budget` 控制和 V2.1 allocation/purity 审计为准。
