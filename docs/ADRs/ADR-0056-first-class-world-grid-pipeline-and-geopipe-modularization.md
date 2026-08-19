# ADR-0056: First-Class World Grid Pipeline and Geopipe Modularization

**Status:** Accepted
**Date:** 2026-08-19

---

## 1. Context

Following the progression of recent architectural improvements across the
geospatial ETL domain in `landseg.geopipe`:
- **PR #67 (ADR-0051)**: Introduced upstream data harmonization to reproject
  and resample disparate remote sensing rasters.
- **PR #68 (ADR-0054)**: Consolidated geospatial ETL under `landseg.geopipe`
  (`harmonize` $\rightarrow$ `ingest` $\rightarrow$ `prepare`) and standardized
  artifact path containers.
- **PR #69 (ADR-0055)**: Established canonical block pooling in `data-ingest`
  and introduced geographic Area-of-Interest (AOI) dataset partitioning in
  `data-prepare`.

Despite these advances, hands-on pipeline workflows revealed remaining
structural friction centered around world grid generation:

1. **Implicit World Grid Generation in `data-harmonize`**:
   World grid construction was embedded as an internal, implicit step of the
   `data-harmonize` pipeline. Users could not generate, inspect, or reuse a
   spatial tiling grid without triggering full raster harmonization.
2. **Dual and Redundant Spatial Contracts (`CanvasSpec` vs. `GridLayout`)**:
   Raster harmonization relied on a bespoke `CanvasSpec` structure, while
   tiling and block assembly relied on `GridLayout`. This redundancy caused
   configuration bloat, duplicated spatial validation logic, and confusion.
3. **Downstream Dependency Coupling**:
   Both `data-ingest` and spatial mapping components require world grid
   specifications. Coupling grid definition exclusively to `data-harmonize`
   made standalone grid workflows and custom grid indexing awkward.
4. **Missing Upstream Pipeline Dependency Enforcement**:
   While `data-ingest` and `data-prepare` validated that their immediate
   upstream reports existed and were marked `SUCCESS`, `data-harmonize` lacked
   an upstream validation check ensuring the world grid artifact existed
   before starting raster warping.

---

## 2. Decision

We elevated World Grid to a first-class execution pipeline (`world-grid`),
modularized `geopipe` grid components, retired `CanvasSpec`, and enforced
upstream pipeline validation before `data-harmonize`.

### 2.1. First-Class `world-grid` Execution Pipeline and API
- **Dedicated Execution Pipeline**:
  Introduced `landseg.execution.pipelines.world_grid` and registered the
  `"world-grid"` command in `landseg.execution.pipelines`.
- **Dedicated API Configurator**:
  Introduced `WorldGridConfigurator` in `landseg.adapters.api`, allowing
  programmatic and notebook-based execution of world grid creation.
- **Hydra Configuration**:
  Created `configs/hydra/pipeline/world-grid.yaml` and
  `configs/hydra/data/world_grid/default.yaml`, and exposed `world-grid:`
  in `configs/user.yaml`.

### 2.2. Geopipe Grid Extraction & Module Boundaries
- **Top-Level Package `landseg.geopipe.grid`**:
  Elevated world grid construction and lifecycle management from
  `geopipe.harmonize.world_grids` to a top-level `landseg.geopipe.grid`
  subpackage (`builder.py`, `lifecycle.py`).
- **Layout Specification in `landseg.geopipe.core`**:
  Consolidated `GridLayout`, `GridMeta`, and `TileIndex` definitions in
  `landseg.geopipe.core.grid_layout`.
- **Elimination of `CanvasSpec`**:
  Retired `CanvasSpec` across harmonization schemas and processors.
  `GridLayout` now serves as the single canonical spatial specification
  defining study extent, CRS, origin, tile size, and tile stride.

### 2.3. Sectional Configuration Independence and Artifact Handshakes
Each pipeline stage is architected to execute using exclusively its local
configuration section:
- **Handshake via Canonical Reports**: Upstream specifications (e.g. grid
  filepath, CRS, raster VRT paths) are persisted into canonical reports
  (`harmonize_report.json`, `ingest_report.json`, `prep_report.json`).
- **Decoupled Downstream Consumption**: Downstream pipelines (such as
  `data-ingest` and `data-prepare`) resolve upstream artifacts directly from
  persisted reports, eliminating the need to redundantly supply or parse
  upstream configuration parameters.

### 2.4. Upstream Pipeline Verification in `executor.py`
Updated `_validate_upstream_pipelines` in `landseg.execution.executor`:
- Recognized `'default'` and `'world-grid'` as standalone entrypoints.
- Enforced that running `'data-harmonize'` verifies the canonical world grid
  artifact (`<output_dpath>/<gid>.json`) exists on disk, raising an
  `artifacts.ArtifactError` if it has not been executed yet.

### 2.5. Interactive Notebook and Test Suite Alignment
- **Notebook 01 Update**:
  Updated `notebooks/01_data_preparation.ipynb` to execute `WorldGridConfigurator`
  in a dedicated step prior to `DataHarmonizationConfigurator`.
- **Streamlined Harmonization API**:
  Simplified `DataHarmonizationConfigurator.set_grid()` to load the existing
  grid artifact directly via `tile_size` and `tile_stride` indexing.
- **End-to-End Integration Testing**:
  Updated `tests/integration/test_end_to_end_pipeline.py` to execute the full
  4-stage pipeline sequence:
  `world-grid` $\rightarrow$ `data-harmonize` $\rightarrow$ `data-ingest` $\rightarrow$ `data-prepare`.

---

## 3. Consequences

### Positive
- **Clear Separation of Concerns**: Spatial extent and tiling layout are
  completely decoupled from raster harmonization and data block assembly.
- **Single Spatial Source of Truth**: `GridLayout` is the sole spatial
  geometry contract across the entire framework.
- **Fail-Fast Safety**: Upstream pipeline validation prevents invalid or
  unaligned harmonization runs when the world grid has not been built.
- **Reusable Artifacts**: Grid JSON artifacts can be shared across multiple
  experiments and downstream tasks without re-running raster warping.

### Negative / Migration
- Pipelines downstream of `world-grid` (CLI or API) now require `world-grid`
  to be run beforehand (or an existing grid JSON artifact supplied).
