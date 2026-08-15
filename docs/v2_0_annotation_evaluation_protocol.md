# V2.0 几何风险标注与评价协议

> 状态：执行版  
> 适用阶段：V2.0 Geometry Risk Validation  
> 目标：以固定、可重复、无循环论证的方式验证 R0 风险图能否预测 A2 的真实局部几何失败。

## 1. 目录约束

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
├─ annotation_working/
│  └─ rigid_structure_centerline/
├─ annotations/
│  ├─ rigid_structure/
│  ├─ soft_stylization/
│  ├─ geometry_failure/
│  └─ uncertainty/
├─ previews/
└─ annotation_manifest.csv
```

`annotations/` 四个目录只允许保存最终 mask，不得放入 RGB 原图、预览图或算法辅助图。人工编辑的 rigid 中心线属于中间产物，单独放在 `annotation_working/`。

## 2. 最终 mask 编码

四类最终标注统一为：

```text
文件格式：PNG
图像模式：8-bit 单通道灰度（PIL mode L）
尺寸：512×512
允许像素值：仅 0 或 255
0   = 不属于该区域
255 = 属于该区域
```

禁止使用 `64/128/192` 等灰度值表达置信度。人工置信度不进入 Ground Truth；无法判断的区域统一进入二值 `uncertainty`。

评价脚本不对最终 mask 做 resize、插值或人工阈值化。模式、尺寸或像素值不符合要求时直接停止。

## 3. 标注单位与空间容忍

V2.0 验证区域级预测能力，不要求逐像素抠边。

- `rigid_structure`：关键几何结构带；
- `soft_stylization`：允许显著风格化的大片软区域；
- `geometry_failure`：错误实际影响的结构区域；
- `uncertainty`：人工无法可靠判断的忽略区域。

除 rigid 外，标注允许自然边界误差，但不得因某张图的结果而任意扩大范围。小于约 `5×5 px` 的孤立错误可以忽略。

## 4. Rigid Structure 固定流程

### 4.1 独立候选

程序仅从对齐后的 content 图生成：

- Canny：阈值 `100/200`；
- LSD：最短线段 `24 px`；
- rigid candidate：Canny 与 LSD 的二值并集。

候选生成代码与 `src/preprocess/structure_risk.py` 分离，不读取 R0 risk map、A2 输出或 failure mask，因此不能直接造成风险图与 GT 的循环一致性。

### 4.2 人工编辑中心线

人工编辑：

```text
annotation_working/rigid_structure_centerline/*.png
```

只判断“哪些线属于关键几何”，主要执行：

1. 删除树叶、草纹、水花、海浪、云层、雪地等软纹理候选；
2. 补充算法漏掉的屋顶、塔尖、建筑外轮廓、道路边界、天际线和关键结构交汇点。

不要把整栋建筑、整片山体或大片墙面内部填白。

### 4.3 固定结构带

人工中心线完成后，程序使用全局固定参数：

```text
dilation radius = 4 px
目标结构带宽约 = 9 px
kernel = 9×9 ellipse
```

生成最终：

```text
annotations/rigid_structure/*.png
```

不允许逐图片改变膨胀半径。评价前会验证最终 rigid mask 是否精确等于冻结参数对中心线的确定性膨胀结果。

### 4.4 场景删除规则

- 建筑：保留外轮廓、屋顶、塔尖、规则门窗、天际线；删除墙面纹理和植被细边。
- 城市街景：保留道路边界、立面、透视主线和天际线。
- 山体：只保留决定场景身份的少数主山脊；删除岩石、树木和雪纹。
- 海浪：候选通常应几乎全部删除，rigid 可以接近空集。
- 天空：通常全部删除。
- 植被：通常删除内部细边，只有决定构图的大树主体轮廓可以保留。

## 5. Soft Stylization

基于 `annotation_sources/content/` 粗填大片软区域，例如：

- 天空、云层；
- 水面、波浪、水花；
- 草地、植被纹理；
- 雪面；
- 不承担关键拓扑的背景和材质区域。

soft 与 rigid 的语义不同，局部重叠应尽量通过人工判断消除，但评价脚本不会强制二者互斥。

## 6. Geometry Failure

必须以 content 与对应 seed 的 A2 输出为依据进行人工标注。标注错误实际影响区域，而不是只描错误边缘：

- 轮廓偏移、断裂或双重轮廓：画有宽度的影响带；
- 新增建筑、三角构筑物或参考对象：填充错误结构整体；
- 屋顶、塔尖、窗格、道路或天际线变形：填充受影响区域；
- 山体、云层、植被建筑化：填充被接管区域；
- 单纯颜色、材质或合理笔触变化：不标。

`helpers/edge_difference/` 只能帮助定位变化，不能自动转为 Ground Truth，也不能替代人工区分“正常风格化”与“错误几何接管”。

## 7. Uncertainty

无法可靠判断是否构成几何失败的区域标为 `255`。典型情况包括某条抽象硬线既可能是合理风格，也可能是参考几何接管。

评价时固定使用：

```text
valid_pixels = uncertainty == 0
```

uncertainty 区域完全排除，不参与正样本、负样本、面积或阈值统计。

## 8. 标注状态与完成确认

程序生成的空 mask 状态为 `pending`；检测到白色像素后为 `in_progress`。空 mask 也可能是合法最终结果，因此程序不依据“是否全黑”自动判断完成。

全部人工复核后必须显式运行完成确认。该命令会验证：

- 所有 source 均 ready；
- 四类最终 mask 均为 512×512、8-bit、严格 `0/255`；
- rigid centerline 已确认；
- rigid final 与固定 4 px 膨胀结果完全一致。

只有随后写入 `complete` 状态的样本可以进入评价。

## 9. 核心评价定义

令：

- `Rτ`：风险图在阈值 `τ` 下的高风险区域；
- `F`：geometry failure mask；
- `Ssoft`：soft stylization mask；
- `Srigid`：rigid structure mask；
- 所有集合均先与 `valid_pixels` 相交。

```text
FailureCoverage = |Rτ ∩ F| / |F|
RiskPrecision   = |Rτ ∩ F| / |Rτ|
FailureIoU      = |Rτ ∩ F| / |Rτ ∪ F|
SoftFPR         = |Rτ ∩ Ssoft| / |Ssoft|
RigidRecall     = |Rτ ∩ Srigid| / |Srigid|
ΔR              = E[R | F=1] - E[R | F=0]
```

同时报告像素级 AUROC、AUPRC、failure prevalence 和 Cohen's d。失败像素稀少时，以 AUPRC 为主要连续指标。

## 10. 阈值约束

不得按图片或配对单独调阈值。所有样本统一报告：

```text
固定阈值：0.30 / 0.50 / 0.70
分位数阈值：top 20% / 35% / 50%
```

分位数只在 `valid_pixels` 上计算。结论不能依赖单一阈值。

## 11. 空集处理

- `F` 为空时，FailureCoverage、AUROC 和 AUPRC 记为 `NaN`，不得伪造为 0 或 1；
- `Rτ` 为空时，RiskPrecision 记为 `NaN`；
- `F` 与 `Rτ` 同时为空时，集合一致性 IoU 记为 `1.0`；
- `Ssoft` 或 `Srigid` 为空时，对应比率记为 `NaN`；
- 汇总均值忽略 `NaN`，并同时报告有效样本数。

## 12. 分组与稳定性

必须分别报告：

- overall；
- 每个配对；
- 每个 seed；
- content rigidity；
- 失败类型 F1/F2/F3/F5/F7（完成类型标注后）。

至少检查 2/3 配对方向是否一致，以及 Demuth × church 三个 seed 中是否至少两个覆盖主要塔尖、屋顶、立面或错误建筑区域。

## 13. V2.1 工程门槛

```text
FailureCoverage >= 0.60
SoftFPR <= 0.35
mean risk in failure > mean risk outside failure
AUPRC > positive-pixel prevalence baseline
```

这些是进入 V2.1 的工程门槛，不是领域通用标准。还必须满足多 seed 方向一致、结论不依赖单一阈值，并确认风险图不是简单的建筑类别 mask。

## 14. 数据泄漏禁令

禁止：

- 使用 R0 risk map 初始化或修改 rigid/failure GT；
- 使用 A2 failure mask 修改 R0；
- 用 edge difference 自动生成 failure GT；
- 每案例调阈值；
- 删除不利案例或只选择有利 seed；
- 将 failure mask 输入生成方法；
- 将当前 Dev 标注集冒充最终 Frozen Test。

R0 评估后最多允许全局修正一次得到 R1，且必须保留完整 R0 结果。

## 15. 执行命令

准备 source、helpers、工作中心线和空 mask：

```bash
python scripts/build_v2_0_annotation_manifest.py
```

人工完成 rigid 中心线后，生成固定 9 px 结构带：

```bash
python scripts/materialize_v2_0_rigid_masks.py --confirm-centerlines-reviewed
```

四类 mask 全部复核后：

```bash
python scripts/finalize_v2_0_annotations.py --confirm-reviewed
python scripts/evaluate_v2_0_geometry_risk.py
python scripts/make_v2_0_geometry_risk_report.py
```
