# V2.3 配对响应画像扩展阶段总结

> 状态：V2.3 已关闭。10 个新增配对完成 seed42 筛查，6 个代表配对完成 seed123/777 多 seed 复核。

## 1. 阶段结论

V2.3 seed42 筛查显示，不同内容—参考图配对对 reference pressure 的响应由三部分组成：

1. 低强度下是否已经出现 takeover；
2. 继续提高 `lambda` 后是否产生新的 takeover；
3. 参考风格是否能够持续转化为有效的 Style。

这三个维度在配对之间组合方式不同，配对交互是当前主要解释变量。

## 2. 人工评分结果

`lambda=.2` 的 takeover 记录为相对 Content 的初始风险，写入 `baseline_takeover_0_3`。`lambda>.2` 记录相对前一档新增风险，写入 `incremental_takeover_0_3`。两类分数分列保存，区分初始失配与 pressure sensitivity。

`compat_G4_city_mismatch` 的目标风格与输出变化不一致，Style 评分记为 `NA`，`style_valid=false`。该配对保留为“低强度即异常失配”样本。

## 3. seed42 配对画像

| 配对 | Style 响应 | 初始风险 | 后续风险 | 阶段分类 |
|---|---|---:|---|---|
| `compat_G1_church` | 强，最高 4 | 0 | 低 | 高效安全型 |
| `clean_kulhanek_G1_water_lake` | 弱，最高 2 | 0 | 无 | 安全但弱风格型 |
| `clean_demuth_G1_water_lake` | 较强，最高 3 | 0 | 后期轻微增加 | 较理想型 |
| `clean_peixotto_G1_water_lake` | 弱，后期出现 | 0 | 无 | 低响应安全型 |
| `clean_kulhanek_G1_forest` | 弱，最高 2 | 0 | 中期持续 | 低效率型 |
| `clean_demuth_G1_forest` | 几乎失败，最高 1 | 0 | 无 | 风格迁移失败型 |
| `compat_G2_opposite_wave` | 强，最高 4 | 1 | 后期加速 | 轻微初始风险、晚期风险型 |
| `compat_G4_sea_cliff_wave` | 强，最高 4 | 2 | 后续趋于饱和 | 高初始风险、饱和型 |
| `compat_G4_city_mismatch` | 无效 | 3 | 无新增 | 低强度异常失配型 |
| `clean_demuth_G4_city_mismatch` | 强，最高 4 | 3 | 早期增加后饱和 | 低强度严重 takeover 型 |

## 4. 关键配对对照

同一内容更换参考图后，Church、water lake 和 city mismatch 均表现出不同的风险与风格响应。Demuth 参考图在 water lake、forest 和 city mismatch 上分别表现为高质量迁移、低响应和低强度严重 takeover。研究对象因此定义为 content-reference pair，分别记录内容效应、参考效应和配对交互。

City mismatch 还显示出另一种状态：部分配对在 `lambda=.2` 已经越过结构安全范围，继续降低普通 reference pressure 的空间有限。此类配对需要后续 controller 提供 reject 或 near-zero reference 分支。

## 5. 阶段决定

V2.3 共完成 10 个新增配对的 seed42 五档扫描，并完成 6 个代表配对的 seed123/777 五档复核。多 seed 结果显示：

- 不同配对具有可区分的 Style 响应、初始 takeover 和增量 takeover 组合；
- `lambda` 增大带来的响应趋势具有配对差异；
- 具体异常的绝对程度和局部变化受 seed 影响；
- `compat_G4_city_mismatch` 属于最低强度即出现严重失配的配对，单独保留 `style_valid=false` 标记。

V2.3 阶段状态：**CLOSED**。

下一阶段进入 V2.4，使用生成前的 content/reference 特征分析响应画像。
