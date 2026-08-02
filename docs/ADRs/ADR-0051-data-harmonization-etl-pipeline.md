# ADR-0051: Introduce Upstream Data Harmonization (ETL) Pipeline

**Status:** Proposed
**Date:** 2026-08-02

---

## 1. Context

The `landseg` framework's data ingestion phase (`data-ingest`) requires input rasters (optical satellite imagery, topographic DEM/DSM layers, LiDAR point cloud metrics, and landcover classification labels) to strictly conform to matching Coordinate Reference Systems (CRS), resolution (pixel size), spatial extent (bounding box), and pixel grid alignment.

In production environments (e.g., Databricks workspaces consuming raw cloud storage inputs), upstream data arrives from diverse data pipelines with spatial inconsistencies:
* Satellite optical composites, LiDAR 20m rasters, and DEM/DSM layers are acquired in different projected CRSs (e.g., WGS84 vs. UTM zones).
* Spatial resolutions vary across providers (e.g., 10m Sentinel, 30m Landsat, 20m LiDAR, 1m DEMs).
* Bounding origins and pixel grid anchors do not snap cleanly to a shared spatial grid.
* Label data often originates from vector shapefiles (`.shp`/`.gpkg`) or raw unaligned rasters.

Requiring users to manually reproject and snap rasters in external GIS applications (such as QGIS or ArcGIS) is unsustainable for automated production MLOps workflows. Furthermore, attempting to inject spatial reprojection directly into the `data-ingest` block assembly layer overcomplicates block building contracts and violates the single-responsibility principle.

---

## 2. Decision

We will introduce an isolated, upstream **Data Harmonization (ETL) Submodule** (`landseg.etl`) and register a dedicated CLI pipeline entry point (`pipeline=data-harmonize`).

### 2.1. Upstream Pipeline Boundary & Pipeline Registration
* A new pipeline flag `pipeline=data-harmonize` will be registered in `landseg.execution.pipelines`.
* `data-harmonize` will execute prior to `data-ingest` in the workflow execution chain. It will transform raw, inconsistent geospatial sources into canonical, aligned rasters anchored to a defined `CanvasSpec`.

### 2.2. Core Submodule Components (`src/landseg/etl/`)
The `landseg.etl` package will be structured into four focused sub-modules:
1. `canvas.py`: Defines `CanvasSpec` (target CRS, target resolution, global spatial extent, and top-left pixel anchor).
2. `warp.py`: Enforces spatial reprojection and pixel grid snapping (`targetAlignedPixels=True`) using Rasterio and GDAL.
   * **Categorical Data (Labels, Domain Masks)**: Will use **Nearest Neighbor** resampling (`Resampling.nearest`) to prevent class label corruption or interpolating invalid class IDs.
   * **Continuous Data (Optical Imagery, DEMs, LiDAR metrics)**: Will use **Bilinear** or **Cubic** resampling.
3. `stacker.py`: Combines separate, aligned single-band or multi-band source rasters into an ordered multi-channel composite raster.
4. `nodata.py`: Unifies disparate upstream nodata flags (`-9999`, `65535`, `0`, `NaN`) into a standardized valid-pixel mask layer.

### 2.3. Virtual Raster (GDAL VRT) Integration
To optimize Databricks cloud storage and compute costs:
* `data-harmonize` will construct lightweight **GDAL Virtual Rasters (`.vrt`)** that encapsulate spatial reprojection and windowed alignment metadata without allocating massive temporary intermediate GeoTIFF files on cloud storage.
* `data-ingest` will read windowed data blocks directly from the generated `.vrt` canvas during block assembly.

### 2.4. Configuration Schema Extension
An `EtlConfig` section will be added to the structured configuration contracts (`src/landseg/configs/schema/sections/etl.py`) and Hydra composition tree, allowing users to configure target CRS, resolution, reference extent, and source path mappings in `configs/user.yaml`.

---

## 3. Consequences

### Positive
* **Production Automation**: Enables zero-manual-GIS automated execution on Databricks from raw cloud storage inputs.
* **Separation of Concerns**: Keeps `data-ingest` focused strictly on tile/block assembly while delegating spatial reprojection to `data-harmonize`.
* **Resource & Storage Efficiency**: Using GDAL VRTs eliminates intermediate disk I/O passes and minimizes cloud storage bloat.
* **Label Integrity**: Enforcing nearest-neighbor resampling for categorical inputs guarantees class ID safety.

### Negative
* **Pipeline Execution Chain Depth**: Adds one additional pipeline stage (`data-harmonize`) prior to `data-ingest`.
* **Dependency Requirement**: Requires GDAL / Rasterio binary bindings in the execution environment.

---

## 4. Implementation Strategy & Modular Roadmap

1. **Phase 1**: Add `EtlConfig` to `src/landseg/configs/schema/sections/` and update Hydra defaults.
2. **Phase 2**: Implement `src/landseg/etl/` (`canvas.py`, `warp.py`, `stacker.py`, `nodata.py`, `orchestrator.py`).
3. **Phase 3**: Register `data-harmonize` in `src/landseg/execution/pipelines/` and add pipeline dependency validation in `executor.py`.
4. **Phase 4**: Add Tier 1 and Tier 2 unit tests under `tests/unit/etl/`.
