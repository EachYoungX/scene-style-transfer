# Image Database Guide

This document is a practical index for collaborators. It explains the image groups used in the project, the typical style behavior of each artist, the commonly used reference/content images, and the failure modes to watch during style transfer.

## Current Dataset Layout

- Style references: `data/raw/<artist>/`
- Content photos: `data/raw/_photo_ref/`
- Clean Artvee style manifest: `data/manifests/clean_artvee_style_refs.csv`
- Clean baseline experiment manifest: `configs/experiment/clean_artvee_baseline_pairs.csv`
- Current active results: `runs/ip_adapter_plus_injection/v1_layer_time_kulhanek_demuth_12step/`
- Archived old runs: `archives/runs_backup_20260714_current.tar.gz`

Do not use `.txt` sidecar descriptions as model inputs. Current experiments use image references plus the CSV `prompt` field.

## Main Clean Artvee Style Groups

### Kulhanek

Role: main clean color reference group.

Visual characteristics:

- Soft watercolor color.
- Warm urban palette.
- Loose architectural edges.
- Relatively mild brushwork compared with Demuth or Van Gogh.
- Useful for testing whether a method can preserve content while adding gentle color and painterly surface.

Canonical reference:

- `data/raw/kulhanek/kulhanek_e18th_street_1956.jpg`

Gallery references:

- `data/raw/kulhanek/kulhanek_cleveland_skyline_1956.jpg`
- `data/raw/kulhanek/kulhanek_reflections_normandy_1971.jpg`
- `data/raw/kulhanek/kulhanek_stutz_last_landscpae_car_1967.jpg`

Common behavior:

- `raw` gives the clearest color/style change.
- `raw` may inject urban objects, roads, buildings, or street-like structure into unrelated content.
- `pooled` is structurally safer but often weak, especially when content/reference elements differ strongly.
- On flower/vegetation content, `pooled` may become gray or muddy instead of producing convincing leaf/flower texture.
- `texture_bank` and `global_plus_texture` should not be used as current solution candidates; IP-Adapter Plus treats patch quilts as small scene/object tokens.

Recommended use:

- Primary style group for V1 layer/time injection experiments.
- Good for first-pass method tuning because failures are visible but not as extreme as Demuth.

### Demuth

Role: high geometric-pressure reference group.

Visual characteristics:

- Precisionist geometry.
- Strong angular composition.
- Torn or fractured geometric planes.
- Harder edges and stronger abstract structure.
- High risk of reference structure takeover.

Canonical reference:

- `data/raw/demuth/demuth_lancaster_1921.jpg`

Gallery references:

- `data/raw/demuth/demuth_masts_1919.jpg`
- `data/raw/demuth/demuth_piano_movers_holiday_1919.jpg`
- `data/raw/demuth/demuth_welcome_to_our_city_1921.jpg`

Common behavior:

- `raw` usually has the strongest style fusion.
- `raw` often introduces objects or geometric elements not present in the content image.
- Structural tearing, extra planes, hard diagonals, and building-like fragments are expected failure signals.
- `pooled` preserves content structure best but loses local material and brush detail.
- Useful for stress-testing whether a method suppresses reference geometry while keeping color/plane style.

Recommended use:

- Use after Kulhanek to test robustness under stronger geometric pressure.
- Do not treat every geometric abstraction as failure. Failure means content identity or topology is changed, such as a new building edge, false road, broken church silhouette, or shifted wave/flower structure.

### Peixotto

Role: linework and grayscale/yellow-paper external generalization group.

Visual characteristics:

- Yellowed paper tone.
- Black ink linework.
- Architectural/city line drawing.
- Strong desaturation pressure.
- More suitable for evaluating line hierarchy than color style score.

Canonical reference:

- `data/raw/peixotto/peixotto_princeton_1897.jpg`

Gallery references:

- `data/raw/peixotto/peixotto_new_york_fort_washington_1897.jpg`
- `data/raw/peixotto/peixotto_philadelphia_2_1897.jpg`
- `data/raw/peixotto/peixotto_philadelphia_view_from_park_1897.jpg`
- `data/raw/peixotto/peixotto_saratoga_schuylers_house_1897.jpg`
- `data/raw/peixotto/peixotto_trenton_old_king_street_1897.jpg`

Common behavior:

- When content and reference subjects are compatible, `raw` can outperform `pooled` because it suppresses original color and creates a faded linework effect.
- When content/reference elements differ strongly, `raw` can impose architectural line structure; `pooled` is safer but weaker.
- Color metrics are misleading for this group. Evaluate structure retention, line hierarchy, and whether the output becomes a coherent drawing.

Recommended use:

- Keep as external generalization after tuning on Kulhanek/Demuth.
- Do not use as the primary tuning group for V1 layer/time schedules.

## Legacy / Diagnostic Style Groups

### Hokusai

Typical references:

- `data/raw/hokusai/hokusai_great_wave_1831.jpg`
- `data/raw/hokusai/hokusai_red_fuji_1831.jpg`
- `data/raw/hokusai/hokusai_shower_summit_1831.jpg`

Use:

- Diagnostic group for strong reference-object leakage.
- Good for wave direction conflict, Fuji/reference-object leakage, signature/watermark contamination, and ukiyo-e line/color pressure.

Known issues:

- Great Wave may leak signature-like details under IP-Adapter Plus raw token injection.
- Reference objects such as Fuji, wave structure, boats, or strong wave direction may enter content.
- Cleaned Hokusai variants were useful diagnostically but should not remain the main experiment path.

Current status:

- Signature/Fuji cleanup branch is archived diagnostically.
- Do not continue no-signature/no-Fuji reruns unless testing data contamination specifically.

### Monet

Typical references:

- `data/raw/monet/monet_pond_water_lilies_1907.jpg`
- `data/raw/monet/monet_garden_giverny_1900.jpg`
- `data/raw/monet/monet_boulevard_capucines_1873.jpg`
- `data/raw/monet/monet_rouen_cathedral_1894.jpg`

Use:

- Impressionist color/atmosphere references.
- Useful for early compatibility and path/topology leakage diagnostics.

Known issues:

- Garden/path references can transfer path topology into flower or forest content.
- Urban references can pressure road perspective and crowd/building layout.

### Van Gogh

Typical references:

- `data/raw/van_gogh/van_gogh_church_auvers_1890.jpg`
- `data/raw/van_gogh/van_gogh_mountains_saint_remy_1889.jpg`
- `data/raw/van_gogh/van_gogh_wheat_field_cypresses_1889.jpg`

Use:

- Strong textured post-impressionist references.
- Useful for testing high texture pressure and architecture deformation.

Known issues:

- Strong style may bend architecture, change roof/edge geometry, or introduce swirled structure.

### Klimt

Typical references:

- `data/raw/klimt/klimt_birch_forest_1903.jpg`
- `data/raw/klimt/klimt_beech_grove_I_1902.jpg`
- `data/raw/klimt/klimt_attersee_1900.jpg`

Use:

- Decorative patterned forest/water references.
- Useful for testing texture transfer in vegetation and soft-structure natural scenes.

Known issues:

- Can replace tree species or alter trunk ordering.
- Pattern pressure may hide paths or depth corridors.

## Common Content Photos

These are the main content images used to evaluate structure preservation and compatibility.

### Core V1 Content Set

- `data/raw/_photo_ref/photo_lecreusois_church.jpg`
  - Case role: `G1_church`.
  - Tests high-rigidity architecture.
  - Watch: roof geometry, facade layout, arches, skyline, false towers/building fragments.

- `data/raw/_photo_ref/photo_sea_wave.jpg`
  - Case role: `G2_opposite_wave`.
  - Tests wave direction and water-motion preservation.
  - Watch: wave direction flips, false mountains, foreign shore/boat objects, copied wave geometry.

- `data/raw/_photo_ref/photo_flower_bed.jpg`
  - Case role: `G3_flower_bed`.
  - Tests vegetation layout without a path.
  - Watch: false paths, tree trunks, garden corridors, flower bed splitting.

- `data/raw/_photo_ref/photo_seregei_city.jpg`
  - Case role: `G4_city_mismatch`.
  - Tests city perspective and high-rigidity urban structure.
  - Watch: road/viewpoint tilt, extra building silhouettes, skyline drift.

### Additional Compatibility / Rigidity Content

- `data/raw/_photo_ref/photo_water_lake_and_boat.jpg`
  - Tests low-rigidity water/horizon scenes.
  - Usually more tolerant of style injection.

- `data/raw/_photo_ref/photo_forest_trees.jpg`
  - Tests forest trunks, path/depth corridor, and high texture density.
  - Watch: path disappearance, tree species replacement, trunk reordering.

- `data/raw/_photo_ref/photo_sea_coast.jpg`
  - Tests low compatibility with wave references and cliff preservation.
  - Watch: cliff replaced by waves, shoreline/topology loss, arbitrary color drift.

- `data/raw/_photo_ref/photo_architecture_basilica.jpg`
  - Tests rigid architecture, towers, facade, and arches.
  - Watch: double boundaries, hybrid roof/tower structures, facade collapse.

- `data/raw/_photo_ref/photo_snow_winter.jpg`
  - Tests street corridor and winter atmosphere.
  - Watch: road boundary drift and false building/path structure.

## Current Method-Relevant Observations

### Raw vs Pooled

`raw` IP-Adapter Plus tokens:

- Strongest visible style.
- Carries useful local style and dangerous reference object/layout information together.
- Commonly causes F3/F5: reference object injection, topology changes, scene regeneration.

`pooled` reference features:

- Safest for content structure.
- Weak local material, brushwork, and line detail.
- Can gray out or flatten incompatible content/reference pairs.

Conclusion:

- The useful local style and dangerous reference semantics are entangled in raw Plus tokens.
- The project should not choose raw or pooled globally.

### Texture Bank / Global Plus Texture

Status: failed diagnostic branch.

Reason:

- Patch quilt/mosaic reference images are still interpreted by IP-Adapter Plus as small objects/scenes.
- Even after fixing black-padding contamination and over-blur, quilt references can inject small buildings, roads, or local scene fragments.

Do not continue:

- `patch quilt`
- `patch mosaic`
- encoded-token shuffle as a final method
- more complex texture-bank image preprocessing as the main path

### Current V1 Direction

Current active hypothesis:

Reference object/layout leakage may be controlled by changing when and where IP-Adapter Plus residuals enter U-Net.

Current active result directory:

- `runs/ip_adapter_plus_injection/v1_layer_time_kulhanek_demuth_12step/`

Current variants:

- `A0_raw_all`: all-layer raw injection baseline.
- `A1_lowres_only`: Down + Mid only; tests whether low-resolution layers cause layout/object leakage.
- `A2_highres_only`: Up only; candidate for texture/detail without early layout takeover.
- `A3_highres_plus_weak_mid`: Up strong + Mid weak + Down off; current V1 candidate.
- `T1_late_style`: adapter off during early denoising, on late.
- `T2_gradual_style`: gradual time schedule.
- `T3_late_highres`: late schedule plus high-resolution layer emphasis.

First visual check suggests:

- `A1_lowres_only` tends to retain more reference structure pressure.
- `A2`, `A3`, and `T3` are the most important candidates to evaluate against `A0_raw_all`.

## Failure Taxonomy for Annotation

Use these labels when writing observations.

- `F1_style_weak`: style is too weak; output is close to content or only lightly filtered.
- `F2_structure_soft_damage`: content structure remains but local details blur, bend, or lose clarity.
- `F3_reference_object_leak`: object from reference appears in output, such as mountain, road, tree trunk, building, signature-like mark.
- `F4_filter_only`: output has minor color/contrast change but no meaningful style.
- `F5_scene_regeneration`: output becomes a new scene or hybrid composition rather than a transfer.
- `F6_double_boundary`: two incompatible geometries are blended, producing ghost edges or invalid structure.
- `F7_material_mismatch`: style changes material/color in an implausible way, such as cliff becoming wave-like or vegetation becoming gray mud.

## Recommended Review Protocol

For each grid, compare in this order:

1. `content`
2. `style`
3. `A0_raw_all`
4. `A2_highres_only`
5. `A3_highres_plus_weak_mid`
6. `T3_late_highres`
7. `pooled` from earlier token-variant runs if needed as the structure-safe lower bound

Record:

- Style strength: weak / medium / strong.
- Structure preservation: low / medium / high.
- Reference leakage: none / minor / clear / severe.
- Best candidate and reason.

For the current V1 decision, the key question is:

`A2`, `A3`, or `T3` must reduce F3/F5 relative to `A0_raw_all` while preserving more style than pooled.

If this holds, proceed to layer-time injection plus spatial rigidity gating.

If this does not hold, proceed to query-token compatibility masking.
