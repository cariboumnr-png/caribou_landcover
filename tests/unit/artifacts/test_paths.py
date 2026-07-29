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

# pylint: disable=protected-access

'''
Unit tests for `landseg.artifacts.paths`.
'''

# standard imports
import os
# local imports
import landseg.artifacts.paths as paths_mod


# ----- `ArtifactPaths` & Foundation tests
def test_artifact_paths_hierarchy() -> None:
    '''verify `ArtifactPaths` namespaces and child path construction.'''
    art = paths_mod.ArtifactPaths(root='/tmp/exp')
    assert art.foundation.root == os.path.join('/tmp/exp', 'foundation')
    assert art.transform.root == os.path.join('/tmp/exp', 'transform')


def test_foundation_paths() -> None:
    '''verify `FoundationPaths` properties and sub-containers.'''
    f = '/tmp/exp/foundation'
    f_paths = paths_mod.FoundationPaths(root=f)
    assert f_paths.report == os.path.join(f, 'ingest_report.json')
    assert f_paths.config == os.path.join(f, 'config.json')

    # world grids fpath formatting
    grid_path = f_paths.grids.fpath((256, 256, 0, 0))
    expected_gid = 'grid_row_256_0_col_256_0.json'
    g_dir = os.path.join(f, 'world_grids')
    assert grid_path == os.path.join(g_dir, expected_gid)

    # domain maps fpaths
    d_dir = os.path.join(f, 'domain_knowledge')
    dom_map = f_paths.domains.domain_map_fpath('eco.tif')
    assert dom_map == os.path.join(d_dir, 'eco.json')

    mapped_tiles = f_paths.domains.mapped_tiles_fpath('eco.tif', 'gid1')
    expected_npz = 'eco_tiles_gid1.npz'
    assert mapped_tiles == os.path.join(d_dir, expected_npz)


def test_data_blocks_paths() -> None:
    '''verify `_DataBlocks` and `_DataBlockPaths` dev/test properties.'''
    db = paths_mod._DataBlocks(root='/tmp/exp/foundation/data_blocks')

    d = '/tmp/exp/foundation/data_blocks/model_dev'
    assert db.dev.blocks == os.path.join(d, 'blocks')
    assert db.dev.catalog == os.path.join(d, 'catalog.json')
    assert db.dev.schema == os.path.join(d, 'schema.json')

    win_path = db.test.mapped_window('g1')
    assert win_path == os.path.join(db.test.windows, 'windows_g1.json')


# ----- `TransformPaths` tests
def test_transform_paths() -> None:
    '''verify `TransformPaths` properties and sub-paths.'''
    t = '/tmp/exp/transform'
    t_paths = paths_mod.TransformPaths(root=t)

    assert t_paths.report == os.path.join(t, 'prep_report.json')
    assert t_paths.config == os.path.join(t, 'config.json')
    assert t_paths.train_blocks == os.path.join(t, 'train_blocks')
    assert t_paths.val_blocks == os.path.join(t, 'val_blocks')
    assert t_paths.test_blocks == os.path.join(t, 'test_blocks')
    assert t_paths.splits_source_blocks == os.path.join(
        t, 'block_splits_source.json'
    )
    assert t_paths.splits_summary == os.path.join(t, 'block_splits_summary.json')
    assert t_paths.label_stats == os.path.join(t, 'label_stats.json')
    assert t_paths.image_stats == os.path.join(t, 'image_stats.json')
    assert t_paths.splits_transformed_blocks == os.path.join(
        t, 'block_splits_transformed.json'
    )
    assert t_paths.schema == os.path.join(t, 'schema.json')


# ----- `ResultsPaths` tests
def test_results_paths_init_and_checkpoints(tmp_path) -> None:
    '''verify `ResultsPaths` initialization, run folder creation, and checkpoint paths.'''
    results = paths_mod.ResultsPaths(results_root=str(tmp_path))
    results.init()

    assert results.run_id == 'run_0001'
    assert os.path.isdir(results.checkpoints)
    assert os.path.isdir(results.logs)
    assert os.path.isdir(results.plots)
    assert os.path.isdir(results.previews)

    c = results.checkpoints
    r = results.run_folder
    assert results.best_checkpoint('model') == os.path.join(c, 'model_best.pt')
    assert results.last_checkpoint('model') == os.path.join(c, 'model_last.pt')
    assert results.phase_status == os.path.join(c, 'status.json')
    assert results.config == os.path.join(r, 'config.json')

    # second run initialization auto-increments run_id
    results_2 = paths_mod.ResultsPaths(results_root=str(tmp_path))
    results_2.init()
    assert results_2.run_id == 'run_0002'

    # trace_to_last option targets previous run
    results_last = paths_mod.ResultsPaths(results_root=str(tmp_path))
    results_last.init(trace_to_last=True)
    assert results_last.run_id == 'run_0002'
