# V2.3 City-Mismatch 配对审计

> 状态：seed42 数据链核查完成，保留为代表性异常配对。

## 1. 审计范围

核查 `compat_G4_city_mismatch` 的异常暖色、远处白色密集树干建筑化，以及 `clean_demuth_G4_city_mismatch` 是否向该配对泄露参考图。审计只使用已有 V2.3 seed42 结果。

## 2. 数据链核查

两条 manifest 记录为：

```text
compat_G4_city_mismatch
  Content:  data/raw/_photo_ref/photo_seregei_street.jpg
  Reference: data/raw/monet/monet_boulevard_capucines_1873.jpg

clean_demuth_G4_city_mismatch
  Content:  data/raw/_photo_ref/photo_seregei_street.jpg
  Reference: data/raw/demuth/demuth_lancaster_1921.jpg
```

两组保存的 `style.png` 与 manifest 参考图经过 `fit_square_crop` 后的结果逐像素一致：`max_diff=0`、`mean_diff=0`。两组共享同一 Content，符合配对设计；输出序列随 `lambda` 增大逐渐分离，参考分支实际生效。

## 3. 颜色行为

在有效 Content 区域使用 `R-B` 作为暖冷变化描述：

| 配对 | λ=.2 | λ=.4 | λ=.6 | λ=.8 | λ=1.0 |
|---|---:|---:|---:|---:|---:|
| Monet compat | 15.2 | 10.5 | 5.7 | 0.9 | -3.8 |
| Demuth clean | 21.5 | 25.4 | 29.6 | 34.0 | 38.0 |

Monet 配对随 reference pressure 增大逐渐变冷，Demuth 配对逐渐变暖，颜色走势与各自参考分支一致。现有证据支持低 λ 暖色来自基础生成或 Content 条件响应；参考图交叉泄露未被发现。

## 4. 视觉结果

`compat_G4_city_mismatch` 存在真实的配对级风险：白色密集树干与街道远景被重组为印象派城市结构，远处树干和侧面块状区域出现建筑化。该现象在低 λ 已出现，并随强度逐步变化。

Demuth 配对表现为更明确的暖色、精确主义平面和角状建筑结构。Monet 配对没有复现 Demuth 的色板与几何特征，两个结果的相似部分来自共享的街景 Content 和城市场景先验。

## 5. 处理决定

1. 数据路径、参考图保存和 runtime instrumentation 均通过核查；
2. `compat_G4_city_mismatch` 保留为“低强度异常失配型”配对；
3. 人工表将其 `style_valid` 设为 `false`，Style 记录为 `NA`；
4. 该配对的后续人工判断聚焦于树干建筑化、相邻 λ 的新增异常和暖色是否属于基础生成偏置；
5. 自动颜色代理用于补充响应描述，人工评分承担该配对的风格判断。
