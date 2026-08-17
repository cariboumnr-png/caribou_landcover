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

# pylint: disable=missing-function-docstring

'''
Catalog adapter utilities.

Provides helpers to load and filter a canonical blocks catalog and schema
to extract class counts and file paths needed for downstream sampling
and analysis.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core

# typing aliases
CatalogDictCtrl = artifacts.Controller[dict[str, geo_core.CatalogEntry]]
SchemaCtrl = artifacts.Controller[geo_core.DataSchema]


class _CatalogViewConfig(typing.Protocol):
    '''Typed configuration container for catalog views.'''
    @property
    def valid_pxs(self) -> dict[str, float]: ...
    @property
    def focal_target(self) -> str | None: ...
    @property
    def non_overlapping_test_grid(self) -> bool: ...


@dataclasses.dataclass(frozen=True)
class DataBlocksView:
    '''High-level view of data blocks for partitioning.'''
    focal_head: str
    base_class_counts: dict[tuple[int, int], list[int]]
    valid_class_counts: dict[tuple[int, int], list[int]]
    blocks: dict[tuple[int, int], str]
    external_test_blocks: list[str] | None = None

    @property
    def dev_base_class_counts(self) -> dict[tuple[int, int], list[int]]:
        '''Backward compatibility property.'''
        return self.base_class_counts

    @property
    def dev_valid_class_counts(self) -> dict[tuple[int, int], list[int]]:
        '''Backward compatibility property.'''
        return self.valid_class_counts

    @property
    def dev_blocks(self) -> dict[tuple[int, int], str]:
        '''Backward compatibility property.'''
        return self.blocks


@dataclasses.dataclass
class _Parsed:
    '''Internal parsed representation of a blocks catalog.'''
    focal_head: str
    base_class_counts: dict[tuple[int, int], list[int]]
    valid_class_counts: dict[tuple[int, int], list[int]]
    valid_file_paths: dict[tuple[int, int], str]


def data_blocks_adapter(
    catalog: str | None = None,
    schema: str | None = None,
    config: _CatalogViewConfig | None = None,
    test_catalog: str | None = None,
    **kwargs: typing.Any,
) -> DataBlocksView:
    '''
    Load and adapt canonical blocks into a structured view for partitioning.

    Filters blocks based on a minimum valid-pixel threshold, derives
    class counts, and optionally incorporates external holdout test blocks.

    Args:
        catalog: Path to canonical blocks catalog JSON.
        schema: Path to dataset schema JSON.
        config: Catalog view configuration.
        test_catalog: Optional path to external holdout catalog JSON.

    Returns:
        DataBlocksView containing filtered metadata for partitioning.
    '''
    effective_catalog = catalog or kwargs.get('dev_catalog')
    effective_schema = schema or kwargs.get('dev_schema')
    effective_test_catalog = test_catalog or kwargs.get('test_catalog')
    effective_config = config or kwargs.get('config')

    assert effective_catalog is not None
    assert effective_schema is not None
    assert effective_config is not None

    # load schema
    data_schema = SchemaCtrl.load_json_or_fail(effective_schema).fetch()
    assert data_schema

    # get block size from schema
    image_shape = data_schema['tensor_shapes']['image']
    blk_size = (image_shape['H'], image_shape['W'])

    # parse catalog
    if effective_config.focal_target:
        assert (
            effective_config.focal_target
            in data_schema['labels']['label_ignore_cls']
        )
    main_parsed = _parse(
        effective_catalog,
        blk_size,
        effective_config.valid_pxs,
        focal_target=effective_config.focal_target
    )

    # optionally parse test data catalog if provided
    test_blocks = None
    if effective_test_catalog:
        try:
            test_parsed = _parse(
                effective_test_catalog,
                blk_size,
                effective_config.valid_pxs,
                focal_target=effective_config.focal_target
            )
            if effective_config.non_overlapping_test_grid:
                test_blocks = list(
                    v for k, v in test_parsed.valid_file_paths.items()
                    if k in test_parsed.base_class_counts
                )
            else:
                test_blocks = list(test_parsed.valid_file_paths.values())
        except artifacts.ArtifactError:
            test_blocks = None

    return DataBlocksView(
        focal_head=main_parsed.focal_head,
        base_class_counts=main_parsed.base_class_counts,
        valid_class_counts=main_parsed.valid_class_counts,
        blocks=main_parsed.valid_file_paths,
        external_test_blocks=test_blocks
    )



def _parse(
    fpath: str,
    block_size: tuple[int, int],
    valid_px_thresholds: dict[str, float],
    *,
    focal_target: str | None = None
) -> _Parsed:
    '''Parse a catalog JSON into filtered class counts and file paths.'''
    catalog_dict = CatalogDictCtrl.load_json_or_fail(fpath).fetch()
    assert catalog_dict
    catalog = geo_core.DataCatalog.from_dict(catalog_dict)

    # fallback to the first target if no focus target is specified
    if not focal_target:
        class_count = next(iter(catalog.values()))['class_count']
        focal_target = next(iter(class_count.keys()))

    # all valid entries from catalog
    work_catalog = {
        k: v for k, v in catalog.items()
        if _is_valid_block(valid_px_thresholds, v['valid_px_ratios'])
    }
    catalog_counts = {
        k: v['class_count'][focal_target] for k, v in work_catalog.items()
    }

    # entries on the base grid (no overlap)
    row_size, col_size = block_size
    base_catalog = {
        k: v for k, v in work_catalog.items()
        if v['row_col'][0] % row_size == 0 and v['row_col'][1] % col_size == 0
    }
    base_counts = {
        k: v['class_count'][focal_target] for k, v in base_catalog.items()
    }

    # all block file paths
    valid_file_paths = {k: v['file_path'] for k, v in work_catalog.items()}

    return _Parsed(
        focal_head=focal_target,
        base_class_counts=base_counts,
        valid_class_counts=catalog_counts,
        valid_file_paths=valid_file_paths
    )


def _is_valid_block(
    valid_thresholds: dict[str, float],
    valid_ratios: dict[str, float]
) -> bool:
    '''Return `True` if all valid thresholds are met.'''
    for k, v in valid_ratios.items():
        threshold = valid_thresholds.get(k)
        if threshold and v < threshold:
            return False
    return True
