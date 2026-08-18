# ADR-0055: Geographic AOI Partitioning, Canonical Block Ingestion, and Dataset Layout Realignment

**Status:** Accepted
**Date:** 2026-08-18

---

## 1. Context

Following the introduction of upstream data harmonization (ADR-0051) and the consolidation of the geospatial ETL domain under `landseg.geopipe` in PR#68 (ADR-0054), hands-on experimentation and end-to-end pipeline evaluations revealed several critical architectural gaps and operational bottlenecks:

1. **Coupled Partitioning Semantics in Upstream Ingestion**:
   Although ADR-0054 organized the pipeline stages into `harmonize`, `ingest`, and `prepare`, data ingestion (`data-ingest`) and harmonization still retained residue of split and partition semantics. Ingested data blocks were coupled to specific split assumptions rather than living in a unified, immutable canonical pool that downstream experiments could partition dynamically.

2. **Ambiguity in World Grid Construction**:
   The canonical world grid (defining tile dimensions, row/column strides, and spatial anchors) was previously rebuilt or referenced ambiguously across both harmonization and ingestion stages. Because raster harmonization defines the target spatial canvas and pixel alignment, defining the canonical world grid belongs naturally to `data-harmonize`.

3. **Absence of Geographic Area-of-Interest (AOI) Constraints**:
   In operational remote sensing and forestry domain applications, model evaluation requires designated, static geographic Areas of Interest (AOIs) (e.g., specific regional study zones, ecological districts, or independently surveyed flight lines) rather than purely stochastic or heuristic scoring splits. The framework lacked the capability for users to supply external geographic rasters to isolate test, validation, or training blocks.

4. **Arbitrary Geometry Overlap and Data Leakage Challenges**:
   Previously, split boundaries and spatial isolation buffers were managed purely on regular, discrete grid indices. Introducing user-supplied AOI rasters introduced new spatial complexity: external rasters can possess arbitrary spatial geometries, irregular boundary contours, and disparate coordinate reference systems (CRSs). Without geometric reprojection, pixel-level intersection masks, and deterministic priority hierarchies, overlapping tiles could inadvertently breach split boundaries or enter multiple splits simultaneously.

5. **Legacy Configuration Artifacts**:
   Post-PR#68 configuration surfaces retained obsolete legacy fields, specifically `dataset_name` (in harmonization and user configs) and `datablocks.name` (in ingestion schemas and Hydra defaults). These fields added unnecessary boilerplate and configuration noise because ingested blocks are uniquely and deterministically identified by spatial index (`row_XXXXXX_col_XXXXXX`) within the experiment directory.

This branch was initiated as a focused follow-up to PR#68 to resolve these architectural deficiencies.

---

## 2. Decision

We established a strictly decoupled geospatial ETL lifecycle where `data-harmonize` defines the canonical world grid and warps raw inputs into harmonized VRTs; `data-ingest` produces a single canonical block pool; and `data-prepare` orchestrates all dataset partitioning, including geographic AOI-driven selection and split isolation.


### 2.1. Pipeline Responsibility Realignment

* **`data-harmonize` (Spatial Normalization & Grid Definition)**:
  Reprojects and warps raw continuous (Sentinel-2, DEM) and categorical (landcover labels, domain masks) rasters to a unified `CanvasSpec`, generates stacked VRT layers, creates valid pixel masks, and constructs the authoritative canonical world grid (`world_grids/grid_row_RRR_col_CCC.json`).

* **`data-ingest` (Canonical Block Assembly)**:
  Consumes canonical harmonized VRTs and the world grid to generate a single, un-partitioned pool of canonical `.npz` data blocks stored under stage `canonical`. All data block assembly occurs once, independent of experimental split configurations.

* **`data-prepare` (Experiment-Level Partitioning & Hydration)**:
  Acts as the sole authority for partitioning canonical data blocks into train, validation, and test splits. Computes split-isolated normalization statistics from training blocks, builds oversampling schemas, and generates runtime hydration catalogs.

### 2.2. Geographic AOI Partitioning and Conflict Resolution
We introduced the `geopipe.prepare.partition.aoi` submodule with `AoiBlockSelector` to support spatial raster-driven block allocation:

* **Spatial Reprojection and Pixel-Level Intersection**:
  AOI rasters are reprojected on the fly into the target canvas CRS. Tile bounding envelopes and active pixel masks are intersected with AOI extents to accurately capture intersecting blocks.

* **Deterministic Priority Hierarchy (`test` > `val` > `train`)**:
  When spatial overlap occurs across user-supplied AOIs, blocks are assigned strictly according to priority: test blocks take precedence over validation blocks, which take precedence over training blocks. Informative warnings are logged when overlaps are detected.

* **Multi-Scenario Partitioning Support**:
  1. *Full AOI Specification*: User supplies `test_aoi`, `val_aoi`, and `train_aoi` rasters.
  2. *Designated Test AOI with Automated Train/Val Splits*: User supplies `test_aoi` to lock static evaluation regions; remaining non-test blocks are automatically partitioned between train and validation using ratio and class scoring criteria.
  3. *Train/Val Only (No Test Holdout)*: User omits `test_aoi` and sets `test_ratio=0.0` (or `test_holdout_blocks_ratio=0.0`), allowing full-training workflows without artificial test holdouts.

* **Spatial Buffer Step Isolation**:
  Preserved spatial buffer rings around test and validation blocks to prevent boundary adjacency leakage.

### 2.3. Test Reference Layout and Synthetic Generator Updates
* Reorganized synthetic test inputs into `input/reference_raster/` hosting `sample_extent.tif` and `sample_test_aoi.tif`.
* Updated `landseg.testing.dummy_data` with an enlarged geographic footprint ($512 \times 768$ canvas @ 20m) to reliably yield 4 training blocks, 1 validation block, and 1 test block selected by the test AOI.

### 2.4. Removal of Legacy `dataset_name` and `datablocks.name`
* Removed `dataset_name` and `datablocks.name` across `configs/user.yaml`, Hydra configuration defaults, dataclass schema definitions, CLI translation mappings, session metadata, and API configurators (`BaseConfigurator` and all child configurators).

---

## 3. Consequences

### Positive
* **Complete ETL Decoupling**: Ingestion produces an immutable canonical block pool once. Multiple experiment runs can partition, score, and balance the same ingested blocks differently without expensive re-ingestion passes.
* **Domain-Rigorous Evaluation**: Enables remote sensing practitioners to enforce fixed geographic holdouts matching real-world study areas.
* **Leakage-Proof Spatial Splits**: Strict CRS-aware geometric intersection, deterministic priority resolution, and spatial buffer rings guarantee zero train/val/test data leakage.
* **Clean Configuration Interface**: Eliminating obsolete dataset naming fields streamlines both `user.yaml` and programmatic API configurator signatures.

### Negative
* **Breaking Changes for Legacy Configs**: Pipelines and scripts passing `dataset_name` or `datablocks.name` will encounter schema validation errors and must remove those fields.

---

## 4. Implementation Summary

1. **Harmonization & Ingestion Refactoring**: Relocated world grid generation to `data-harmonize` and unified `data-ingest` into a single canonical block writer.

2. **Geographic AOI Partitioning**: Implemented `AoiBlockSelector` in `geopipe.prepare.partition.aoi` supporting full AOIs, test-only AOIs, and train/val only modes with priority conflict resolution.

3. **Synthetic Generator Alignment**: Updated `dummy_data.py` to generate enlarged continuous/categorical rasters and reference rasters under `reference_raster/`.

4. **Config Surface Purge**: Removed `dataset_name` and `datablocks.name` across schemas, Hydra defaults, `user.yaml`, CLI translators, configurators, and pipelines.

5. **Quality Assurance**: Verified all unit and integration test suites across adapters, configs, and pipelines, and confirmed successful end-to-end execution of `data-harmonize`, `data-ingest`, `data-prepare`, and `diagnose-overfit`.
