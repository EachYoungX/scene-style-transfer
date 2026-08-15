# V2.0 几何风险标注与评价协议

> 状态：执行版  
> 适用阶段：V2.0 Geometry Risk Validation  
> 目标：以固定、可重复、无循环论证的方式验证 R0 风险图能否预测 A2 的真实局部几何失败。

## 1. 目录契约

```text
data/derived/v2_0_geometry_risk/
├─ annotation_sources/
│  ├─ content/
│  ├─ a2_outputs/
│  └─ helpers/
│     ├─ canny/
│     ├─ lsd/
│     ├─ rigid_candidates/
│     └─ edge_difference/
├─ valid_masks/
│  ├─ valid_content/
│  └─ valid_eval/
├─ annotations/
│  ├─ rigid_structure/
│  ├─ soft_stylization/
│  ├─ geometry_failure/
│  └─ uncertainty/
├─ previews/
└─ annotation_manifest.csv
```

`annotations/` 四个目录只保存最终 mask，不得放入 RGB 原图、预览图或算法辅助图。

## 2. 最终 mask 编码

四类人工标注统一为：

```text
格式：PNG
模式：8-bit 单通道灰度（PIL mode L）
尺寸：512×512
允许值：仅 0 或 255
0   = 不属于
255 = 属于
```

禁止使用 `64/128/192` 等灰度表达置信度。无法判断的区域进入二值 `uncertainty`。评价脚本不 resize、不插值、不阈值化人工 mask；发现错误模式、尺寸或灰度值时直接停止。

## 3. 自动有效区域

### 3.1 valid_content

`valid_content_mask` 由对齐预处理自动生成：

```text
255 = 原始图像等比缩放后的真实内容
0   = 补成 512×512 时加入的黑色 padding
```

padding 不是场景结构，不得人工标成 rigid、soft、failure 或 uncertainty。

### 3.2 valid_eval

padding/content 接缝可能含插值混合像素和人工高梯度，因此固定使用：

```text
valid_eval = erode(valid_content, radius=2 px)
```

所有空间指标、连续风险统计和分位数阈值只在 `valid_eval` 内计算。人工 uncertainty 继续从中排除：

```text
valid_pixels = valid_eval & (uncertainty == 0)
```

## 4. Rigid Structure Edge Mask

`rigid_structure` 的正式含义是：

> 内容图中对场景几何身份重要的刚性边缘和结构线。

它是细线 mask，不是整物体区域，也不是膨胀后的结构带。

```text
普通边缘：1 px
离散化跨两行/两列：允许 2 px
极少数模糊但明确的硬边：最多 3 px
```

不得统一膨胀到 9 px。窗框、屋檐、墙角和道路边界在粗膨胀后会连接成面，失去结构线语义。

### 4.1 独立候选

程序从 content 自动生成：

- Canny：阈值 `100/200`；
- LSD：最短线段 `24 px`；
- rigid candidate：Canny 与 LSD 的二值并集。

所有候选在输出前执行：

```text
candidate &= valid_eval
```

因此 padding 边缘和接缝不会进入候选。辅助代码不读取 R0 risk map、A2 输出或 failure mask，避免风险图与 GT 的循环论证。

### 4.2 人工规则

人工查看 content 和 rigid candidate，在空白 `annotations/rigid_structure/*.png` 上绘制最终细线：

- 建筑：保留外轮廓、屋顶、塔尖、规则门窗、天际线；删除墙面纹理和植被细边。
- 城市街景：保留道路边界、立面、透视主线和天际线。
- 山体：只保留决定场景身份的少数主山脊；删除岩石、树木和雪纹。
- 海浪：高梯度波峰、水花和泡沫不是刚性结构，rigid 通常接近空集。
- 天空：通常为空。
- 植被：通常删除内部细边，仅保留极重要的大树主体轮廓。

必须区分“高梯度”和“几何刚性”：Canny 可以检测波峰、云层和树叶，但这些通常不属于 rigid GT。

## 5. Soft Stylization

基于 content 粗填允许显著风格化的大片区域，包括天空、云层、水面、波浪、水花、草地、植被纹理、雪面以及不承担关键拓扑的背景材质。

最终 soft 与 rigid 必须严格不重叠。完成确认时采用固定规则：

```text
rigid_guard = dilate(rigid, radius=1 px)
soft_final = soft_manual & valid_content & ~rigid_guard
```

rigid 优先；1 px guard 只用于 soft 去重，不会改写 rigid GT。

## 6. Geometry Failure

每个 A2 seed 独立标注。必须以 content 与对应 A2 输出为依据，标记错误实际影响范围：

- 轮廓偏移、断裂、错误硬线或双重轮廓：细线或窄影响区域；
- 新增建筑、三角构筑物或参考对象：错误结构整体；
- 屋顶、塔尖、窗格、道路或天际线变形：受影响区域；
- 山体、云层、植被建筑化：被接管区域；
- 单纯颜色、材质或合理笔触变化：不标。

`helpers/edge_difference/` 只能帮助定位变化，不能自动生成 GT，也不能替代人工判断正常风格化与错误结构接管。

## 7. Uncertainty

`uncertainty` 只表达 geometry failure 判断的不确定区域：

```text
255 = 无法可靠判断是否构成几何失败
```

它不是模糊的 failure 置信度，也不用于 rigid/soft 的边界纠结。评价时 uncertainty 像素直接排除。

## 8. 完成确认与自动规范化

程序生成的空 mask 状态为 `pending`；出现白色像素后为 `in_progress`。合法最终 mask 也可能全黑，因此不能按面积自动判定完成。

人工复核后显式执行完成确认。程序会：

1. 验证四类 mask 均为 512×512、8-bit、严格 `0/255`；
2. 将四类 mask 裁到 `valid_content`，自动移除 padding；
3. 按 1 px rigid guard 清除 soft 重叠；
4. 验证最终 `rigid & soft == 0`；
5. 将清单状态写为 `complete`。

只有 `complete` 样本可以进入评价。

## 9. 空间容差原则

人工 GT 保持原始细线。空间容差只在评价时临时计算，不写回任何 GT 文件。

冻结参数：

```text
spatial tolerance radius = 2 px
```

除精确指标外，额外报告：

```text
FailureCoverage@2px = |dilate(Rτ, 2) ∩ F| / |F|
RiskPrecision@2px   = |Rτ ∩ dilate(F, 2)| / |Rτ|
RigidRecall@2px     = |dilate(Rτ, 2) ∩ Srigid| / |Srigid|
```

膨胀结果始终再与 `valid_pixels` 相交，不能跨入 padding 或 uncertainty。精确指标继续保留，便于判断结果是否完全依赖容差。

## 10. 核心评价定义

令 `Rτ` 为阈值后的高风险区域、`F` 为 failure、`Ssoft` 为 soft、`Srigid` 为 thin rigid edge；所有集合先与 `valid_pixels` 相交。

```text
FailureCoverage = |Rτ ∩ F| / |F|
RiskPrecision   = |Rτ ∩ F| / |Rτ|
FailureIoU      = |Rτ ∩ F| / |Rτ ∪ F|
SoftFPR         = |Rτ ∩ Ssoft| / |Ssoft|
RigidRecall     = |Rτ ∩ Srigid| / |Srigid|
ΔR              = E[R | F=1] - E[R | F=0]
```

同时报告 AUROC、AUPRC、failure prevalence、Cohen's d 和上述 2 px 容差指标。失败像素稀少时，以 AUPRC 为主要连续指标。

## 11. 阈值约束

不得按图片、seed 或配对单独调阈值。全部样本统一报告：

```text
固定阈值：0.30 / 0.50 / 0.70
分位数阈值：top 20% / 35% / 50%
```

分位数只在 `valid_pixels` 上计算。结论不能依赖单一阈值或只依赖容差版本。

## 12. 空集处理

- `F` 为空：FailureCoverage、AUROC、AUPRC 为 `NaN`；
- `Rτ` 为空：RiskPrecision 为 `NaN`；
- `F` 与 `Rτ` 同时为空：集合一致性 IoU 为 `1.0`；
- `Ssoft` 或 `Srigid` 为空：对应比率为 `NaN`；
- 汇总均值忽略 `NaN`，同时报告有效样本数。

不得把未定义指标伪造为 0 或 1。

## 13. 分组和工程门槛

必须报告 overall、每配对、每 seed、content rigidity，以及完成类型标注后的 F1/F2/F3/F5/F7。

进入 V2.1 的工程参考门槛：

```text
FailureCoverage >= 0.60
SoftFPR <= 0.35
mean risk in failure > mean risk outside failure
AUPRC > positive-pixel prevalence baseline
```

还必须满足：至少 2/3 配对方向一致；Demuth × church 至少两个 seed 覆盖主要结构失败；结论不依赖单一阈值或只有 tolerant 指标通过；风险图不是简单建筑类别 mask。

## 14. 数据泄漏禁令

禁止：

- 使用 R0 risk map 初始化或修改 rigid/failure GT；
- 使用 A2 failure mask 修改 R0；
- 用 edge difference 自动生成 failure GT；
- 每案例调阈值；
- 删除不利案例或选择有利 seed；
- 将 failure mask 输入生成方法；
- 将当前 Dev 标注集冒充最终 Frozen Test。

R0 评估后最多允许一次全局修正得到 R1，并必须保留完整 R0 结果。

## 15. 执行命令

准备 source、helpers、valid masks 和空白 annotations：

```bash
python scripts/build_v2_0_annotation_manifest.py
```

四类 mask 全部人工复核后：

```bash
python scripts/finalize_v2_0_annotations.py --confirm-reviewed
python scripts/evaluate_v2_0_geometry_risk.py
python scripts/make_v2_0_geometry_risk_report.py
```
