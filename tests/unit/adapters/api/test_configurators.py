# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''Unit tests for API configurator classes.'''

# local imports
import landseg.adapters.api.configurators as configurators


# ----- `DataHarmonizationConfigurator` tests
def test_data_harmonization_configurator(tmp_path):
    '''
    Given: Parameters for data harmonization ETL.
    When: Chaining methods on `DataHarmonizationConfigurator`.
    Then: Correctly populate underlying `RootConfig` and validate.
    '''
    ref_tif = tmp_path / 'ref.tif'
    ref_tif.write_text('ref')
    manifest_json = tmp_path / 'manifest.json'
    manifest_json.write_text('[]')

    cfg_builder = configurators.DataHarmonizationConfigurator(
        experiment_root=str(tmp_path),
        dataset_name='test_ds'
    )
    cfg_builder.set_canvas(
        target_crs='EPSG:3161',
        target_resolution=20.0,
        reference_raster=str(ref_tif)
    ).set_grid(
        tile_size=512,
        tile_overlap=64,
        crs='EPSG:3161'
    ).set_dataset_manifest(
        dataset_manifest=str(manifest_json),
        dataset_name='test_ds'
    ).set_resampling(
        continuous='bilinear',
        categorical='nearest'
    ).set_output_dpath(
        output_dpath=str(tmp_path / 'harmonized')
    )

    root = cfg_builder.running_root_config
    assert root.pipeline.name == 'data-harmonize'
    assert root.data.harmonization.canvas.target_crs == 'EPSG:3161'
    assert root.data.harmonization.canvas.target_resolution == 20.0
    assert root.data.harmonization.canvas.reference_raster == str(ref_tif)
    assert root.data.harmonization.grid.tile_specs.size_row == 512
    assert root.data.harmonization.grid.tile_specs.size_col == 512
    assert root.data.harmonization.grid.tile_specs.overlap_row == 64
    assert root.data.harmonization.grid.tile_specs.overlap_col == 64
    assert root.data.harmonization.grid.crs == 'EPSG:3161'
    assert root.data.harmonization.dataset_manifest == str(manifest_json)
    assert root.data.harmonization.dataset_name == 'test_ds'
    assert root.data.harmonization.resampling_continuous == 'bilinear'
    assert root.data.harmonization.resampling_categorical == 'nearest'
    assert root.data.harmonization.output_dpath == str(tmp_path / 'harmonized')


# ----- `DataIngestionConfigurator` tests
def test_data_ingestion_configurator(tmp_path):
    '''
    Given: Parameters for data ingestion pipeline.
    When: Chaining methods on `DataIngestionConfigurator`.
    Then: Correctly populate ingestion rebuild and harmonization run fields.
    '''
    cfg_builder = configurators.DataIngestionConfigurator(
        experiment_root=str(tmp_path),
        dataset_name='test_ds'
    )
    cfg_builder.set_rebuild(
        rebuild=True
    ).set_harmonization_run(
        target_run=1
    )

    root = cfg_builder.running_root_config
    assert root.pipeline.name == 'data-ingest'
    assert root.data.ingestion.rebuild is True
    assert root.data.ingestion.harmonization_run == 1



# ----- `DataPreparationConfigurator` tests
def test_data_preparation_configurator(tmp_path):
    '''
    Given: Parameters for data preparation pipeline.
    When: Chaining methods on `DataPreparationConfigurator`.
    Then: Correctly populate partition and oversampling settings.
    '''
    cfg_builder = configurators.DataPreparationConfigurator(
        experiment_root=str(tmp_path),
        dataset_name='test_ds'
    )
    cfg_builder.set_partition(
        validation_blocks_ratio=0.15,
        test_holdout_blocks_ratio=0.05
    ).set_oversampling(
        target_head='class_head',
        reward_classes={1: 2.0}
    ).set_rebuild(
        rebuild=True
    )

    root = cfg_builder.running_root_config
    assert root.pipeline.name == 'data-prepare'
    assert root.data.preparation.partition.val_ratio == 0.15
    assert root.data.preparation.partition.test_ratio == 0.05
    assert root.data.preparation.catalog.focal_target == 'class_head'
    assert root.data.preparation.scoring.reward == {1: 2.0}
    assert root.data.preparation.rebuild is True


# ----- `TrainingSessionConfigurator` tests
def test_training_session_configurator(tmp_path):
    '''
    Given: Parameters for model training session.
    When: Chaining methods on `TrainingSessionConfigurator`.
    Then: Correctly populate model, optimizer, data loader, and tasks.
    '''
    cfg_builder = configurators.TrainingSessionConfigurator(
        experiment_root=str(tmp_path),
        dataset_name='test_ds'
    )
    cfg_builder.set_model(
        body='unet',
        bottleneck='conv',
        base_channel=64
    ).set_optimization(
        optimizer='AdamW',
        learning_rate=1e-3,
        weight_decay=1e-4,
        scheduler='CosAnneal'
    ).set_data_loading(
        batch_size=16,
        patch_size=256
    ).set_domain_source(
        category_domain='sample_domain_1',
        continuous_domain=None
    ).set_tasks(
        logit_adjust_alpha=0.5,
        exclude_classes={'class_head': [255]},
        loss_weights={'ce': 1.0},
        head_weights={'class_head': 1.0}
    ).set_runtime(
        max_epochs=10,
        active_heads=['class_head'],
        patience_epoch=5,
        track_heads={'class_head': 1.0}
    )

    root = cfg_builder.running_root_config
    assert root.pipeline.name == 'model-train'
    assert root.models.model_body == 'unet'
    assert root.models.bottleneck == 'conv'
    assert root.models.model_body_registry['unet'].base_ch == 64
    assert root.session.engine_optim.opt_cls == 'AdamW'
    assert root.session.engine_optim.lr == 1e-3
    assert root.session.data_loader.batch_size == 16
    assert root.session.data_loader.patch_size == 256
    assert root.data.specification.domain_ids_name == 'sample_domain_1'
