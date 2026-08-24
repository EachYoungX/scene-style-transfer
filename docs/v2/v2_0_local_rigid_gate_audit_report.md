# V2.0 局部刚性门控审计报告

> 状态：**已关闭**
> 
> Snow: **regional/coarse-support failure candidate**  
> Church: **edge-local failure candidate**

## 审计对象

- case：`Kulhanek × snow_winter`
- seed：`123`
- schedule：`A2_highres_only`
- retention：`rho=0.0`
- ROI：`x=205..231, y=312..334`（512×512 坐标，诊断用）
- 目标 rigid 连通分量：`x=209..227, y=316..329`

## A–E：局部门控审计

审计通过。

- annotation source 与实际 runtime content 完全一致：`max_abs_pixel_diff=0`、`mean_abs_pixel_diff=0`。
- 目标 ROI 内保留 rigid 像素：`59`。
- 实际 active spatial resolutions：`64×64`、`32×32`、`16×16`。
- 所有 active resolution 的 rigid-related token 均无漏映射。
- 同一次 forward 的 `raw_ip_delta → gated_ip_delta`：
  - rigid token RMS ratio：`0.0`；
  - non-rigid token RMS ratio：`1.0`。
- 3 个 resolution、所有 active processor/timestep 的局部检查均通过。

因此可以确认：

> snow 中心小建筑的 rigid mask 已在实际 A2 high-resolution IP-Adapter image branch 中正确生效。

证据目录：

```text
runs/ip_adapter_plus_injection/v2_0_local_gate_audit/
└─ v1_5_kulhanek_snow_winter/seed123/audit_snow_seed123/
   ├─ input_alignment/
   ├─ effective_gates/
   ├─ local_residual/
   ├─ overlays/
   └─ audit_summary.md
```

## F：输出空间响应

edge-only gate 已产生可观测输出变化，但没有把最终中心建筑轮廓锁定在人工 rigid 线上。

报告：

```text
runs/ip_adapter_plus_injection/v2_0_local_gate_audit/
└─ v1_5_kulhanek_snow_winter/seed123/spatial_response/
```

## G：filled-region 最小干预

将中心建筑 ROI 外扩 2 px 并整体填充为诊断 rigid mask，仍使用 seed123、rho=0、A2 schedule。

结果：

- edge-only ROI RGB diff mean vs Uniform：`24.2652`
- filled-region ROI RGB diff mean vs Uniform：`24.4536`
- edge-only 与 filled-region 的 ROI 平均差异：约 `6.8422`

整体视觉轮廓没有出现足以支持“filled region 能锁定建筑几何”的变化，edge-only 支持范围过窄不足以解释该结果。

证据目录：

```text
runs/ip_adapter_plus_injection/v2_0_local_gate_audit/
└─ v1_5_kulhanek_snow_winter/seed123/
   ├─ diagnostic_masks/
   └─ filled_region_probe/
```

## H：分辨率因果探针

分别只在一个 resolution 上施加 gate：

| Gate resolution | 目标 rigid RMS ratio | 其他 resolution ratio | ROI diff vs Uniform |
|---:|---:|---:|---:|
| 64×64 | 0.0 | 1.0 | 5.4450 |
| 32×32 | 0.0 | 1.0 | 13.8266 |
| 16×16 | 0.0 | 1.0 | 24.3360 |

16×16 对最终局部响应最强，接近全 gate 结果；但单独压制 16×16 仍不等于几何边界锁定。

证据目录：

```text
runs/ip_adapter_plus_injection/v2_0_local_gate_resolution_probe/
```

## 最终结论

本轮排除了：

- annotation source/runtime content 坐标错位；
- valid_eval 删除目标 rigid；
- 512→token-grid 映射漏掉细线；
- active processor 未使用 gate；
- gate broadcasting 错误；
- timestep 或 resolution 未实际执行 gate；
- non-rigid 区域被意外削弱。

当前 snow 结果应归类为：

> 局部 rigid residual gate 实现正确，且在 16×16 等实际 attention token 上确实生效；但该 IP-Adapter image-branch residual 对 snow 中心建筑最终几何的控制权不足，尚未形成可靠的几何边界保护。

下一阶段保留当前 gate 的实现和审计证据，坐标与 token 映射审计到此结束。后续几何保护实验转向 denoising/noise-space 或结构分支。
