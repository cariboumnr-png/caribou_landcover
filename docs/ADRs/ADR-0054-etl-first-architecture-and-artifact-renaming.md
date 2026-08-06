# ADR-0054: Establish Unified Geospatial ETL Architecture in `geopipe` and Codebase-Wide Naming Realignment

**Status:** Accepted
**Date:** 2026-08-06

---

## 1. Context

During the recent evolution of the `landseg` framework, the spatial processing pipeline underwent significant structural changes. Originally, data ingestion (`data-ingest`) and data preparation (`data-prepare`) performed ad-hoc preprocessing, while `data-harmonize` was introduced as an auxiliary step. This led to several architectural issues:
1. **Ambiguous Pipeline Semantics**: The term "ETL" was used interchangeably for the entire data pipeline, for individual sub-modules, or for raw raster harmonization, leading to confusion between data ingestion, dataset partitioning, and spatial warping.
2. **Fragmented Spatial Processing**: Geospatial Extract, Transform, and Load (ETL) logic was scattered across standalone packages and pipeline scripts rather than living under a single, unified geospatial domain module.
3. **Legacy Naming & Container Discrepancies**: Data artifact containers and path classes retained legacy names (`ETLPaths`, `FoundationPaths`, `TransformPaths`, `ResultsPaths`) that did not correspond to the actual pipeline stages (`data-harmonize`, `data-ingest`, `data-prepare`, `session`).
4. **Implicit & Rigid Run Folder Resolution**: Downstream pipelines (`data-ingest`) implicitly resolved the latest harmonization run folder without allowing users to explicitly select a targeted run folder when multiple harmonization experiments existed.

To resolve these ambiguities and clean up technical debt, a comprehensive architectural realignment was executed across the codebase.

---

## 2. Decision

We established a **Unified Geospatial ETL Architecture** consolidated entirely within the `landseg.geopipe` domain module and executed a codebase-wide terminology and naming realignment.

### 2.1. Unified Geospatial ETL Domain (`landseg.geopipe`)
All geospatial Extract, Transform, and Load operations—spanning raw raster warping, tiling grid construction, domain mapping, block generation, and split partitioning—are now unified under `landseg.geopipe` in three distinct pipeline submodules:
1. `geopipe.harmonize` (Data Harmonization Stage): Ingests raw continuous (Sentinel-2, DEM) and categorical (landcover labels, domain masks) rasters, warps/reprojects them to a defined spatial canvas, and outputs canonical VRTs in run-isolated directories (`harmonized_data/run_XXXX/`).
2. `geopipe.ingest` (Data Ingestion Stage): Consumes canonical VRTs from `geopipe.harmonize`, builds world grids and domain knowledge maps, tiles data into raw `.npz` data blocks, and registers catalogs/schemas (`ingested_data/`).
3. `geopipe.prepare` (Data Preparation Stage): Partitions data blocks into train/validation/test splits, computes normalization statistics, applies class balancing/oversampling, and outputs prepared data artifacts (`prepared_data/`).


### 2.2. Standardized Artifact Path Dataclass Hierarchy
We renamed and refactored artifact path containers across `src/landseg/artifacts/paths/` and top-level lazy exports in `src/landseg/artifacts/__init__.py`:
* `HarmonizationPaths` (formerly `ETLPaths`): Manages VRT outputs and reports for `data-harmonize`.
* `IngestionPaths` (formerly `FoundationPaths`): Manages world grids, domain maps, and data blocks for `data-ingest`.
* `PreparationPaths` (formerly `TransformPaths`): Manages block splits, catalogs, and schemas for `data-prepare`.
* `SessionPaths` (formerly `ResultsPaths`): Manages checkpoints, metrics, and logs for model execution.
* Decoupled path sub-containers using `typing.Protocol` interfaces to prevent tight coupling between execution modules and concrete filesystem path classes.

### 2.3. Flexible Targeted Harmonization Run Selector (`harmonization_run`)
We added a configuration knob `harmonization_run` under `data.ingestion` (and exposed it in `configs/user.yaml` under `data-ingest:` and in `DataIngestionConfigurator` as `.set_harmonization_run()`).
* Accepts: `int` run index (e.g. `1`), numeric `str` (e.g. `"1"`), folder name `str` (e.g. `"run_0001"`), or direct directory path `str`.
* Defaults to `None` (resolving the latest run folder if unspecified).

### 2.4. Codebase-Wide Naming, Docstring, and Documentation Synchronization
* Replaced all legacy terms (`foundation`, `transform`, `results` as pipeline/artifact names) across python source files, docstrings, unit tests, Hydra YAML config trees, and documentation files (`ARCHITECTURE.md`, `project_structure.md`, `project_structure_fr.md`, `workflow_chart.md`, `user.yaml`).
* Synchronized `user.yaml` default configuration overrides with the Hydra schema defaults.
* Enforced Ontario Crown Copyright header compliance across all project files.

---

## 3. Consequences

### Positive
* **Unified Domain Architecture**: `geopipe` serves as the single home for all 3 geospatial ETL stages (`harmonize` $\rightarrow$ `ingest` $\rightarrow$ `prepare`), providing a cohesive, self-contained spatial processing pipeline.
* **Clean Code & Namespace Compliance**: Class names match module names and pipeline stages (`HarmonizationPaths`, `IngestionPaths`, `PreparationPaths`, `SessionPaths`), eliminating developer confusion.
* **Targeted Ingestion Execution**: Users can explicitly pinpoint previous harmonization runs when ingesting datasets without re-running harmonization.
* **Decoupled Protocols**: Using `typing.Protocol` interfaces for path containers facilitates unit testing and mocking.
* **Synchronized Documentation & Configs**: Hydra trees, `user.yaml`, architecture diagrams, and French/English docs are 100% aligned.

### Negative
* **Breaking Changes for Legacy Code**: Scripts or external configs referencing `ETLPaths`, `FoundationPaths`, `TransformPaths`, or `ResultsPaths` must update imports to the new class names.

---

## 4. Implementation Summary

1. **Phase 1**: Consolidated geospatial ETL pipelines (`harmonize`, `ingest`, `prepare`) under `landseg.geopipe`.
2. **Phase 2**: Renamed artifact path dataclasses (`HarmonizationPaths`, `IngestionPaths`, `PreparationPaths`, `SessionPaths`) and updated lazy imports in `landseg.artifacts`.
3. **Phase 3**: Decoupled path containers via `typing.Protocol` interfaces in `geopipe`.
4. **Phase 4**: Added `harmonization_run` selector across schema, Hydra defaults, `user.yaml`, `translate.py`, API configurators, and pipelines.
5. **Phase 5**: Codebase-wide docstring, comment, and documentation update (`project_structure.md`, `workflow_chart.md`, `user.yaml`, `README.md`).
6. **Phase 6**: Updated and added unit test suites (`test_paths.py`, `test_config_root.py`, `test_translate.py`, `test_configurators.py`, `test_data_ingest.py`).
