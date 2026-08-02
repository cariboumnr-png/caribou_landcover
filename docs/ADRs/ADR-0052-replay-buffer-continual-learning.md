# ADR-0052: Introduce Replay Buffer Layer for Continual Model Updating

**Status:** Proposed
**Date:** 2026-08-02

---

## 1. Context

In production landcover segmentation deployments, datasets arrive across two distinct temporal cycles:
1. **Static / Slow-Changing Base Layers**: High-resolution LiDAR DEM/DSM and SGM rasters acquired periodically (e.g., 5-year flyover cycles across geographic units).
2. **Dynamic / Fast-Changing Layers**: Satellite optical composites (Landsat/Sentinel/Planet) arriving continuously (monthly or seasonally).

When updating models on newly arrived satellite imagery or new geographic regions, naive fine-tuning exclusively on new data causes **catastrophic forgetting**. The neural network's weights overwrite previously learned spatial, spectral, and topographic features, leading to severe performance degradation on historical regions or different seasons.

Conversely, retraining models from scratch across all cumulative historical datasets for every minor satellite update is computationally expensive and impractical for continuous production ML pipelines on Databricks.

---

## 2. Decision

We will introduce a **Two-Tier Training Architecture** backed by a **Replay Buffer Layer** to support continuous model updates without catastrophic forgetting.

### 2.1. Two-Tier Training System Strategy
* **Tier 1: Base Foundation Model Training**:
  * Periodically (e.g., annually or upon receiving major new 5-year LiDAR flight units), a comprehensive Base Model will be trained across a multi-regional, multi-temporal snapshot of all historical data.
  * The resulting checkpoint will be logged to the MLflow Model Registry as the foundational weight initialization baseline (`base_model.pt`).
* **Tier 2: Continuous Model Fine-Tuning**:
  * When new satellite composites or localized label updates arrive, a fine-tuning run will be initialized from the Tier 1 Base Model weights.
  * Fine-tuning will use a lower learning rate and a **Replay Buffer** to preserve historical knowledge.

### 2.2. Replay Buffer Implementation (`landseg.session.data`)
A dedicated `ReplayBatchSampler` will be introduced inside `src/landseg/session/data/sampler.py`:
* **Mini-Batch Ratio Control**: Ensures every mini-batch during training contains a strict, configurable proportion of new tiles (e.g., 80%) mixed with historical "anchor" tiles (e.g., 20%) sampled from previous dataset manifests.
* **Anchor Sampling**: Historical anchor tiles will be sampled across diverse geographic units and seasons to maintain invariant feature representations (topographic relationships, cloud/shadow masks, and seasonal reflectance range).

### 2.3. DataSpec & Dataset Assembly Support (`pipeline=data-prepare`)
* The data preparation pipeline (`data-prepare`) will be updated to accept an optional `replay_manifest_archive` configuration parameter.
* When assembling dataset splits for model updating runs, `data-prepare` will combine the current dataset manifest with sampled entries from historical manifest archives, tagging tiles with `stream` metadata (`current` vs. `replay`).

### 2.4. Session Checkpoint Weight Initialization
* `landseg.session` configuration will be updated to explicitly support initializing model weights from a specified local checkpoint path or MLflow Registry URI (`model.init_weights`), allowing seamless fine-tuning starting points.

---

## 3. Consequences

### Positive
* **Catastrophic Forgetting Protection**: Retains model accuracy on historical geographical units and past seasons while adapting to new incoming satellite imagery.
* **Compute & Time Efficiency**: Dramatically reduces training time and Databricks GPU compute costs compared to retraining from scratch on cumulative historical datasets.
* **Deterministic Provenance**: Replay sampling ratios and historical tile manifest hashes will be recorded in `DataSpec` artifacts for full auditability.

### Negative
* **Storage Requirement for Replay Pool**: Requires maintaining an indexed archive of historical data block manifests and sample tiles.
* **Configurability Complexity**: Introduces hyperparameter tuning for `replay_ratio` (defaulting to 0.2 / 20%).

---

## 4. Implementation Strategy & Modular Roadmap

1. **Phase 1**: Add `ReplayBatchSampler` in `src/landseg/session/data/sampler.py`.
2. **Phase 2**: Update `data-prepare` pipeline to support merging current manifests with historical replay manifest pools.
3. **Phase 3**: Extend `landseg.session` model factory to handle checkpoint weight initialization (`init_weights`).
4. **Phase 4**: Add Tier 1 and Tier 2 unit tests under `tests/unit/session/data/` verifying mini-batch sampling ratio guarantees.
