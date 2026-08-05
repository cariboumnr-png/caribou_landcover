# Raster Data Preparation Guide

[English](./data_preparation.md) | [Français](./data_preparation_fr.md)

Last updated: 2026-08-02

---

## Overview

Effective geospatial machine‑learning workflows begin with careful and consistent data preparation. Before any modeling, training, or experimentation can take place, all input rasters must be standardized in terms of CRS, resolution, extent, and alignment so they behave predictably throughout the pipeline. Ensuring these core properties are harmonized at the start greatly reduces downstream complexity and prevents errors related to mismatched spatial reference systems or misaligned pixels.

A central part of this standardization is the definition of a static spatial extent as the canvas for tiling and deterministic indexing. Once the extent is established, all datasets intended for training, inference, or future production must be snapped or reprojected to match its CRS and pixel size. Users can prepare rasters externally using GIS tools such as QGIS, ArcGIS, or GDAL (see the tutorial below), or use the project's built-in `data-harmonize` ETL pipeline (`python scripts/run.py pipeline=data-harmonize`) to reproject and composite raw GeoTIFFs automatically.

This guide outlines the recommended steps for creating a reference raster, enforcing consistent alignment across datasets, and exporting clean, grid‑compatible rasters that form a reliable foundation for all subsequent analysis and modeling.

---

## Contents

- [World Grid Definition](#world-grid-definition)
- [Input Data Raster Specification](#input-data-raster-specification)
  - [Image Raster](#image-raster)
  - [Label Raster](#label-raster)
  - [Domain Raster (Optional)](#domain-raster-optional)
- [Raster Alignment Requirements](#raster-alignment-requirements)
- [Data Configuration JSON](#data-configuration-json)
- [Project Folder Structure Layout](#project-folder-structure-layout)

---

## World Grid Definition

The world grid must be defined in a projected CRS so that tile coordinates and pixel dimensions correspond to linear units (e.g., metres). The starting point is to establish the **project extent**—the full area enclosing both the training region and all intended prediction regions. This extent should be considered **immutable for the entire project**, ensuring that all subsequent data preparation steps reference the same spatial domain.

Once the extent is fixed, users may generate one or more **world grids** as stable, versioned artifacts for different experimental needs. For example, grids may vary by tile size (affecting model field‑of‑view), or may include/omit tile overlap to study edge effects. While grids may change across experiments, they must all remain anchored to the same CRS, resolution, and origin defined by the project extent. This project assumes the rasters are always anchored at **top-left**.

The grid extent can be provided in two ways:

  - Manual definition using a top‑left origin and a specified number of tiles in the horizontal and vertical directions.
  - Reference‑raster definition (preferred), where a raster created in common GIS tools (QGIS, ArcGIS, GDAL) supplies the CRS, pixel resolution, extent, and origin to build grid.

After project extent is defined, world grids are derived from it during the pipeline (module `landseg.geopipe.ingest.world_grids`) to form reproducible, versioned tiling schemes used throughout experimentation and production.

[Jump](#tutorial---create-a-reference-raster) to the tutorial on how to create reference raster in QGIS.

<img src="./images/extent_reference.png" alt="extent_reference" width="800">

**Figure 1**. Extent reference raster creation.

---

## Input Data Raster Specification

### Image Raster
Image rasters used for model training and prediction typically originate from satellite platforms such as *Landsat*, accessed either through the [USGS EarthExplorer portal](https://earthexplorer.usgs.gov/) or through [Google Earth Engine (GEE)](https://earthengine.google.com/). You may choose whichever workflow you are more comfortable with.
>**Note:** scene selection, mosaicking, cloud masking, and other QA/QC decisions remain outside the scope of this framework, as they depend heavily on project‑specific requirements and user expertise.

For GEE users, we recommend exploring the **Best Available Pixel (BAP)** workflow, which provides flexible tools for assembling high‑quality annual composites. A widely adopted implementation is available here: <https://github.com/saveriofrancini/bap>. BAP‑style compositing helps produce temporally stable, cloud‑free rasters suitable for downstream ML models.

The input image composite is flexible: users may supply any arbitrary set of raster channels, and there are no rigid band count or band ordering requirements. Optional derived features—such as spectral indices (e.g., NDVI, NDWI) or topographic layers derived from a DEM (e.g., slope, aspect)—are computed automatically by downstream pipelines *only if* the required optical bands or elevation layer are mapped in `image_band_map`. Users can supply as many or as few channels as their application requires.

<img src="./images/example_image_raster.png" alt="example_image_raster" width="800">

**Figure 2**. Example image raster.

---

### Label Raster
Label rasters are fully user‑defined, as the labeling system originates from the user’s domain knowledge, data sources, and project objectives. This framework does not prescribe any specific classification scheme; instead, it expects users to supply a raster containing the land‑cover or segmentation labels relevant to their workflow.

Because the project is designed for land‑cover segmentation, the label raster should contain:

  - `Integer` class IDs, representing land‑cover categories.
  - A clearly defined `NoData` value.
  - Any classes the user intends to ignore during training (e.g., water, cloud, unclassified areas).

During data preparation, both `NoData` and user‑specified ignore‑classes are automatically converted into a single ignore‑label index (commonly 255, user‑configurable). This ensures clean handling of invalid or unwanted pixels throughout training and inference.

In many real‑world classification systems, the number of raw land‑cover classes can be large, imbalanced, or difficult to model effectively in a single pass. To support more manageable and staged training strategies, this framework provides an optional two‑tier parent–child label hierarchy:

  - Parent classes represent broader, generalized groups.
  - Child classes represent the finer‑scale raw categories that belong to each parent group.

This hierarchy enables workflows such as:

  1. Training an initial model on coarse parent groups to learn broad structure.
  2. Refining the model by focusing on selected parent groups and training on the full child classes associated with them.

If you wish to use this hierarchical approach, you must provide a JSON configuration that defines the parent–child mappings. The format and usage of this configuration are described later in the guide.

<img src="./images/example_label_raster.png" alt="example_label_raster" width="800">

**Figure 3**. Example label raster.

---

### Domain Raster (Optional)
A domain raster is an ***optional*** input that can be included when the study benefits from specifying ecological, geographic, or management sub‑regions. The domain can represent any user‑defined partitioning relevant to the project—eco‑zones, administrative boundaries, disturbance regimes, biophysical strata, or other contextual divisions. Although optional for training, a domain raster should ideally **cover both the training region and the intended prediction area** to ensure consistent conditioning across the full project extent.

The domain raster must be **integer‑valued**, with each integer representing a unique domain category. Users do not need to pre‑process these values beyond ensuring their correctness; during training, the framework automatically converts the raw domain raster into the internal representations required by the chosen conditioning strategy (whether used as concatenated inputs, FiLM‑style conditioning, or left as discrete indices).

Because domain processing occurs within the training configuration—not the data‑preparation stage—this guide only requires users to provide a clean, integer‑encoded domain raster aligned to the project extent and reference raster.

<img src="./images/example_domain_raster.png" alt="example_domain_raster" width="800">

**Figure 4**. Example domain raster.

---

## Raster Alignment Requirements

All input rasters—image, label, and optional domain—must be **aligned** to the project’s reference raster created during world‑grid definition. This ensures that every raster shares:

- **The same projected CRS**
- **The same pixel resolution**
- **The same pixel origin and alignment**

Snapping to the reference raster guarantees that pixel boundaries match exactly, which is essential for deterministic tiling, correct label–image pairing, and reproducible experiments.

All rasters must also fall **entirely within the bounds** of the project extent. Any data extending beyond the reference raster’s extent will be clipped or discarded during alignment. Users should therefore crop or reproject their data appropriately before entering the pipeline.

[Jump](#tutorial---snapping-workflow-in-qgis) to the tutorial on how to align rasters to a reference raster in QGIS.

---

## Data Configuration JSON

A data configuration JSON accompanies the input rasters. It defines band ordering, label specifications, and optional class remapping.

### Core Configuration Fields
| Key | Purpose | Notes |
|-----|---------|-------|
| `image_band_map` | Defines channel band order of the composite image | Must be 0‑based index mapping |
| `label_specs` | Defines target task head label specifications | Contains `num_cls`, `ignore_cls`, and `reclass_map` |

**Example:**
```json
{
  "image_band_map": {
    "dem": 0, "blue": 1, "green": 2,
    "red": 3, "nir": 4, "swir1": 5, "swir2": 6
  },
  "label_specs": {
    "main_task": {
      "num_cls": 8,
      "ignore_cls": [0, 255],
      "reclass_map": {
        "1": 1, "2": 1,
        "3": 2, "4": 2,
        "5": 3, "6": 3
      }
    }
  }
}
```

### Optional Metadata Fields
These improve interpretability and visualization, but are not required by the preprocessing pipeline.

| Key | Purpose |
|-----|---------|
| `label_class_name` | Human‑readable names for raw label categories |
| `label_reclass_name` | Human‑readable names for parent classes |
| `label_reclass_color_map` | RGB color arrays for class preview visualization |

**Key Rules**
- Band indices in `image_band_map` must be 0‑based.
- `ignore_cls` values (such as `0` or `255`) are automatically handled during data ingestion.
- `reclass_map` is optional; use only if leveraging parent–child class grouping.

---

## Project Folder Structure Layout

Input rasters, generated pipeline artifacts, and session execution results are organized within a structured experiment root directory (`<exp_root>`).

For the complete, detailed directory tree layout across inputs, artifacts, and training results, see:
- [Experiment & Artifacts Directory Tree Layout](./experiment_directory_layout.md) ([Français](./experiment_directory_layout_fr.md))

> **Synthetic Mock Data Generation**: To populate a local testing environment with a complete set of synthetic sample GeoTIFFs, run:

> ```bash
> python scripts/generate_dummy_data.py
> ```

---

## Tutorial - Create a Reference Raster

### Step 1 — Select Projected CRS
1. Open QGIS.
2. In the bottom‑right corner, click the CRS button.
3. Select your local projected system (e.g., `EPSG:3161`).

---

### Step 2 — Create Extent Layer
1. Go to **Layer → Create Layer → New Shapefile Layer**.
2. Set Geometry Type to **Polygon**.
3. Draw a bounding polygon that fully covers:
   - All training imagery
   - All expected prediction regions
4. Save the shapefile as `project_extent.shp`.

---

### Step 3 — Rasterize the Extent
1. Go to **Raster → Conversion → Rasterize (Vector to Raster)**.
2. Select `project_extent.shp` as the input.
3. Set the fixed value to `1`.
4. Choose the target resolution (e.g., `20` metres).
5. Output format: **GeoTIFF**.
6. Save as `reference_extent.tif`.

All world grids and snapped input rasters will be anchored to this reference.

---

### Tutorial - Snapping Workflow in QGIS
With the reference extent raster prepared, all remaining input rasters (image, label, and optional domain) must be reprojected and snapped to match it. The following example workflow uses QGIS:

---

**Task 1 — Load Data**<br>
1. Open QGIS.
2. Drag in:
   - The **reference extent raster**.
   - Your **image raster**.
   - Your **label raster**.
   - Your **domain raster** (optional).

---

**Task 2 — Open the Align Rasters Tool**<br>
1. Open the **Processing Toolbox**.
2. Navigate to: **GDAL → Raster Alignment → Align rasters**.

---

**Task 3 — Set Up Alignment Parameters**<br>
1. **Input layer:** Select the raster you wish to align (image, label, or domain).
2. **Reference layer:** Choose the **reference extent raster**.
3. **Output raster size:**
   - Target resolution: **Layer resolution** (inherits reference pixel size)
   - Target CRS: automatically taken from the reference raster
4. **Output alignment:**
   - Enable **Match pixel alignment**
   - Enable **Clip to reference layer extent**

---

**Task 4 — Save the Aligned Raster**<br>
Save as `image_aligned.tif`, `labels_aligned.tif`, and `domain_aligned.tif`.

---

**Task 5 — Repeat for All Rasters**<br>
Run the alignment process for each input raster individually.

**Result**<br>
All rasters now share the same CRS, pixel resolution, origin, and spatial bounds, fully compatible with the world grid and project pipelines.