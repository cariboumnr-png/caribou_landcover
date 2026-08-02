# Artifacts & Results Directory Tree Layout

This directory defines the canonical filesystem path structures for all dataset artifacts and execution results produced by the `landseg` pipelines.

```text
<exp_root>/
│
├── artifacts/
│   │
│   ├── harmonized/                              # Data Harmonization (ETL) Outputs
│   │   ├── harmonized_<name>.vrt                # Reprojected single-source Virtual Raster
│   │   ├── harmonized_image_composite.vrt       # Canonical multi-channel stacked feature Virtual Raster
│   │   ├── valid_pixel_mask.vrt                 # 1-band boolean valid-pixel mask Virtual Raster
│   │   └── etl_report.json                      # ETL execution summary report
│   │
│   ├── foundation/                              # Data Ingestion Outputs
│   │   ├── world_grids/
│   │   │   └── grid_row_<srow>_<orow>_col_<scol>_<ocol>.json *
│   │   ├── domain_knowledge/
│   │   │   ├── <domain_name>.json *
│   │   │   └── <domain_name>_tiles_<gid>.npz
│   │   ├── data_blocks/
│   │   │   ├── model_dev/
│   │   │   │   ├── blocks/
│   │   │   │   ├── windows/
│   │   │   │   │   └── windows_<gid>.json
│   │   │   │   ├── catalog.json
│   │   │   │   └── schema.json
│   │   │   └── test_holdout/
│   │   │       ├── blocks/
│   │   │       ├── windows/
│   │   │       │   └── windows_<gid>.json
│   │   │       ├── catalog.json
│   │   │       └── schema.json
│   │   ├── ingest_report.json                   # Ingestion execution summary report
│   │   └── config.json                          # Ingestion configuration record
│   │
│   └── transform/                               # Data Preparation Outputs
│       ├── train_blocks/                        # Transformed training array blocks
│       ├── val_blocks/                          # Transformed validation array blocks
│       ├── test_blocks/                         # Transformed testing array blocks
│       ├── block_splits_source.json
│       ├── block_splits_summary.json
│       ├── block_splits_transformed.json
│       ├── label_stats.json
│       ├── image_stats.json
│       ├── schema.json
│       ├── prep_report.json                     # Preparation execution summary report
│       └── config.json                          # Preparation configuration record
│
└── results/                                     # Model Training & Evaluation Results
    └── run_0001/                                # Serialized experiment run directory
        ├── checkpoints/
        │   ├── status.json                      # Training phase status tracking
        │   ├── <name>_best.pt                   # Best model checkpoint weights
        │   └── <name>_last.pt                   # Final model checkpoint weights
        ├── logs/                                # Training run execution logs
        ├── plots/                               # Metric and diagnostic visualization plots
        ├── previews/                            # Model output prediction previews
        ├── config.json                          # Fully resolved execution configuration
        ├── evaluation.json                      # Evaluation metrics JSON (if evaluation run)
        ├── summary.json                         # Overall run summary metrics JSON
        └── step_results.json                    # Per-step loss/metric tracking JSON

* Note: Sidecar `_meta.json` files are generated alongside grid and domain JSON artifacts.
```
