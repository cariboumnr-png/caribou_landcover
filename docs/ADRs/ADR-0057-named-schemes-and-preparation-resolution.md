# ADR-0057: Named Schemes and Data Preparation Resolution

**Status:** Accepted — Implemented
**Date:** 2026-09-02

---

## 1. Context

In previous versions of the `landseg` geospatial ETL and preparation pipeline:

1. **Contaminated & Static Sidecar Metadata**:
   Dataset metadata sidecar JSON files mixed intrinsic physical raster
   properties (such as index base, class counts, and colormaps) with fixed
   experiment-level target reclassifications (`reclass` and `reclass_name`),
   locking the dataset into a single static hierarchy during harmonization
   and ingestion.
2. **All-or-Nothing Feature Channels**:
   Ingestion and normalization operated indiscriminately on all channels of
   raw feature rasters without allowing users to configure or slice specific
   feature bands (e.g., RGB vs. RGB-NIR vs. Surface Reflectance) for downstream
   experiments.
3. **Semantic Proximity vs. Experiment Ergonomics**:
   While experiment-level selection belongs in `user.yaml` or notebooks,
   writing raw band names or complex class mappings from scratch for every
   experiment configuration was tedious and severed the connection to
   reusable domain semantics established alongside the data sources.

---

## 2. Decision

We decoupled intrinsic raster metadata from experiment-level data preparation
by introducing **Named Schemes** and a dedicated **Preparation Resolution
Layer** across `landseg.geopipe`.

### 2.1. Pure Categorical Specifications and Unified Index Base
- **Purified `CategoricalSpecs`**:
  Stripped `reclass` and `reclass_name` out of `CategoricalSpecs` (renamed from
  `LabelSpecs`) so that it strictly describes physical GeoTIFF raster properties
  (`index_base`, `num_cls`, `ignore_cls`, `class_name`, `color_map`, `taxonomy`).
- **Explicit `index_base` Support**:
  Unified 0-based and 1-based raster indexing across label and domain
  categorical rasters, enforcing strict validation against class index bounds.

### 2.2. Named Schemes in Dataset Sidecars
- **Manifest Sidecar Schema (`schemes`)**:
  Sidecars now declare optional, semantically close named schemes:
  - **`FeatureSchemes`**: Named band groupings (e.g.,
    `rgb: ["blue", "green", "red"]`, `rgb_nir: ["blue", "green", "red", "nir"]`).
  - **`LabelSchemes`**: Named hierarchical target reclassifications (e.g.,
    `binary: {reclass: {"1": [1, 2], "2": [3]}, reclass_name: {"1": "forest", "2": "water"}}`).
  - **Domain Rasters**: Strictly enforced to have `schemes: null`.
- **Harmonized VRT Tagging & Ingestion Schema Propagation**:
  - `data-harmonize`: Writes resolved schemes into the harmonized VRT dataset
    metadata tags (`schemes={cfg['name']: schemes}`).
  - `data-ingest`: Reads embedded schemes from the source VRTs via
    `io.read_schemes()` and records them into the dataset `schema.json`
    artifact under `dataset.schemes`.

### 2.3. Dedicated Preparation Resolution Layer (`geopipe.prepare.resolver`)
- **Module Separation**:
  Introduced `landseg.geopipe.prepare.resolver` housing pure metadata and
  configuration resolution:
  - `resolve_feature_channels()`: Resolves active band names and 0-based
    channel indices against `image_band_map` using `raster_schemes` or inline
    band lists.
  - `resolve_target_reclass()`: Resolves active multi-head target hierarchies
    against `label_names` using `raster_schemes` or inline reclass
    dictionaries per label layer.
- **Pipeline Consumption (`data_prepare.py`)**:
  `data-prepare` loads `schema.json`, extracts `raster_schemes`, and passes
  them to the resolver functions before slicing and normalizing data blocks.

### 2.4. Selective Statistics Aggregation and Dynamic Normalization
- **Channel-Specific Image Statistics (`stats.py`)**:
  Updated global Welford stats aggregation to compute per-band normalization
  statistics exclusively on the selected feature channels.
- **Dynamic Multi-Head Target Construction (`normalize.py`)**:
  Block normalization slices images to active channels and generates
  multi-head target label stacks via `_reclassify_label_stack` on the fly,
  preserving the immutability of canonical ingested `.npz` data blocks.

### 2.5. Dual Public Interfaces (CLI & Programmatic API)
- **CLI Configuration (`configs/user.yaml` & Hydra)**:
  Exposed `features` and `targets` sections under `data-prepare:`, mapped
  into `data.preparation` via `landseg.adapters.cli.translate`.
- **Programmatic Python API (`DataPreparationConfigurator`)**:
  Added fluent `set_features()` and `set_targets()` methods for notebook and
  script environments.

---

## 3. Consequences & Benefits

- **Semantic Proximity**: Domain experts define reusable band combinations
  and reclassification hierarchies directly alongside the raw dataset sidecars.
- **Experiment Agility**: Data scientists can test different feature subsets
  and target definitions in `user.yaml` or notebooks without re-running data
  harmonization or ingestion.
- **Immutability of Ingested Blocks**: Ingested `.npz` blocks remain pure,
  canonical representations of the underlying geospatial layers.
- **Single Responsibility & Cohesion**: Clear separation between metadata
  resolution (`resolver.py`), catalog filtering (`adapter.py`), and tensor
  operations (`normal_blocks/`).
