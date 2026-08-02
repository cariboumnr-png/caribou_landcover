# ADR-0051: Introduce Upstream Data Harmonization (ETL) Pipeline

**Status:** Accepted
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

We introduced an isolated, upstream **Data Harmonization (ETL) Submodule** (`landseg.etl`) and registered a dedicated CLI pipeline entry point (`pipeline=data-harmonize`).

### 2.1. Upstream Pipeline Boundary & Pipeline Registration
* A new pipeline flag `pipeline=data-harmonize` was registered in `landseg.execution.pipelines`.
* `data-harmonize` executes prior to `data-ingest` in the workflow execution chain. It transforms raw, inconsistent geospatial sources into canonical, aligned rasters anchored to a defined `CanvasSpec`.

### 2.2. Core Submodule Components (`src/landseg/etl/`)
The `landseg.etl` package was structured into focused sub-modules:
1. `canvas.py` / `spatial.py`: Defines `CanvasSpec` and canvas resolution helper `create_canvas` (target CRS, target resolution, global spatial extent, and top-left pixel anchor).
2. `warp.py`: Enforces spatial reprojection and pixel grid snapping (`targetAlignedPixels=True`) using Rasterio and GDAL.
   * **Categorical Data (Labels, Domain Masks)**: Uses **Nearest Neighbor** resampling (`Resampling.nearest`) to prevent class label corruption or interpolating invalid class IDs.
   * **Continuous Data (Optical Imagery, DEMs, LiDAR metrics)**: Uses **Bilinear** or **Cubic** resampling.
3. `stacker.py`: Combines separate, aligned single-band or multi-band source rasters into an ordered multi-channel composite raster.
4. `nodata.py`: Unifies disparate upstream nodata flags (`-9999`, `65535`, `0`, `NaN`) into a standardized valid-pixel mask layer.

### 2.3. Virtual Raster (GDAL VRT) Integration
To optimize Databricks cloud storage and compute costs:
* `data-harmonize` constructs lightweight **GDAL Virtual Rasters (`.vrt`)** that encapsulate spatial reprojection and windowed alignment metadata without allocating massive temporary intermediate GeoTIFF files on cloud storage.
* `data-ingest` reads windowed data blocks directly from the generated `.vrt` canvas during block assembly.

### 2.4. Configuration Schema Extension
An `EtlConfig` section was added to the structured configuration contracts (`src/landseg/configs/schema/sections/etl.py`) and Hydra composition tree, allowing users to configure target CRS, resolution, reference extent, and source path mappings in `configs/user.yaml`.

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

1. **Phase 1**: Added `EtlConfig` to `src/landseg/configs/schema/sections/` and updated Hydra defaults.
2. **Phase 2**: Implemented `src/landseg/etl/` (`canvas.py`, `warp.py`, `stacker.py`, `nodata.py`, `spatial.py`).
3. **Phase 3**: Registered `data-harmonize` in `src/landseg/execution/pipelines/` and added pipeline orchestration in `data_harmonize.py`.
4. **Phase 4**: Added unit tests under `tests/unit/etl/`.
