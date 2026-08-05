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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''
Data config schema
'''

# standard imports
import dataclasses
import re
# local imports
import landseg.configs.schema.utils as utils

# alias
field = dataclasses.field


# ----- data harmonization
@dataclasses.dataclass
class _Canvas:
    reference_raster: str = ''
    target_crs: str | None = None
    target_resolution: float | None = None


@dataclasses.dataclass
class _RawData:
    dev_features: dict[str, str] = field(default_factory=dict)
    domains: dict[str, str] = field(default_factory=dict)
    dev_labels: dict[str, str] = field(default_factory=dict)
    test_features: dict[str, str] = field(default_factory=dict)
    test_labels: dict[str, str] = field(default_factory=dict)


@dataclasses.dataclass
class _HarmonizationCfg:
    canvas: _Canvas = field(default_factory=_Canvas)
    raw_data: _RawData = field(default_factory=_RawData)
    dataset_name: str = 'sample_data'
    dataset_config: str = ''
    resampling_continuous: str = 'bilinear'
    resampling_categorical: str = 'nearest'
    output_dpath: str = 'experiment/artifacts/harmonized_data'

    def validate(self) -> None:
        utils.must_exist(self.canvas.reference_raster, 'Reference raster')
        if self.dataset_config:
            utils.must_exist(self.dataset_config, 'Dataset configuration JSON')

        if (
            self.canvas.target_crs and
            not bool(re.fullmatch(r'epsg:\d+', self.canvas.target_crs, re.I))
        ):
            raise ValueError('Invalid CRS identifier. Must be "EPSG:...."')

        if (
            self.canvas.target_resolution and
            self.canvas.target_resolution <= 0.0
        ):
            raise ValueError('target_resolution must be positive.')


# ----- data ingestion
@dataclasses.dataclass
class _Extent:
    origin: tuple[float, float] = (0.0, 0.0)
    pixel_size: tuple[float, float] = (0.0, 0.0)
    grid_extent: tuple[float, float] | None = None
    grid_shape: tuple[int, int] | None = None


@dataclasses.dataclass
class _TileSpecs:
    size_row: int = 256
    size_col: int = 256
    overlap_row: int = 0
    overlap_col: int = 0

    def validate(self):
        # current we only accept equal row and col sizes and strides
        if self.size_row != self.size_col:
            raise ValueError('Only square blocks are supported.')

        if self.overlap_row != self.overlap_col:
            raise ValueError('Only equal row/column stride is supported.')

        if self.size_row <= 0:
            raise ValueError('Block size must be positive.')

        if self.overlap_row < 0:
            raise ValueError('Block stride must be zero or positive.')


@dataclasses.dataclass
class _Grid:
    mode: str = 'ref'
    crs: str = ''
    extent: _Extent = field(default_factory=_Extent)
    tile_specs: _TileSpecs = field(default_factory=_TileSpecs)

    @property
    def tile_specs_tuple(self) -> tuple[int, int, int, int]:
        '''Tile specs in px as (row, col, overlap_row, overlap_col).'''
        return dataclasses.astuple(self.tile_specs)

    def validate(self) -> None:
        if self.mode != 'ref':
            raise ValueError(
                f'Invalid grid mode: {self.mode}. '
                'Data ingestion requires "ref" grid mode mandated by '
                'data harmonization.'
            )
        # crs string format (optional if derived from ref raster)
        if self.crs and not bool(re.fullmatch(r'epsg:\d+', self.crs, re.I)):
            raise ValueError('Invalid CRS identifier. Must be [EPSG:....]')
        # tile specs
        self.tile_specs.validate()


@dataclasses.dataclass
class _Domains:
    valid_threshold: float = 0.7
    target_variance: float = 0.9

    def validate(self) -> None:
        pass


@dataclasses.dataclass
class _DataBlocks:
    name: str = ''
    ignore_index: int = 255
    image_dem_pad: int = 8

    def validate(self) -> None:
        pass


@dataclasses.dataclass
class _IngestionCfg:
    grid: _Grid = field(default_factory=_Grid)
    domains: _Domains = field(default_factory=_Domains)
    datablocks: _DataBlocks = field(default_factory=_DataBlocks)
    rebuild: bool = False
    output_dpath: str = '${execution.exp_root}/artifacts/ingested_data'

    def validate(self) -> None:
        self.grid.validate()
        self.domains.validate()
        self.datablocks.validate()


# ----- data preparation
@dataclasses.dataclass
class _CatalogView:
    valid_pxs: dict[str, float] = field(default_factory=lambda: {'image': 0.9})
    focal_target: str | None = None
    non_overlapping_test_grid: bool = True

    def validate(self):
        for k, v in self.valid_pxs.items():
            utils.must_within(v, f'{k} valid threshold', 0, 1)

@dataclasses.dataclass
class _Partition:
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    buffer_step: int = 1

    def validate(self):
        utils.must_within(self.val_ratio, 'validation block ratio', 0, 1)
        utils.must_within(self.test_ratio, 'test holdout block ratio', 0, 1)

@dataclasses.dataclass
class _Scoring:
    reward: dict[int, float] = field(default_factory=dict)
    alpha: float = 1.0
    beta: float = 0.0

    def validate(self):
        utils.must_within(self.alpha, 'scoring alpha', 0)
        utils.must_within(self.beta, 'scoring beta', 0)

@dataclasses.dataclass
class _Hydration:
    max_skew_rate: float = 10.0

    def validate(self):
        utils.must_within(self.max_skew_rate, 'hydration skew ratio', 0)

@dataclasses.dataclass
class _PreparationCfg:
    catalog: _CatalogView = field(default_factory=_CatalogView)
    partition: _Partition = field(default_factory=_Partition)
    scoring: _Scoring = field(default_factory=_Scoring)
    hydration: _Hydration = field(default_factory=_Hydration)
    rebuild: bool = False
    output_dpath: str = '${execution.exp_root}/artifacts/prepared_data'

    def validate(self):
        self.catalog.validate()
        self.partition.validate()
        self.scoring.validate()
        self.hydration.validate()


# ----- data specs
@dataclasses.dataclass
class _Specification:
    domain_ids_name: str | None = None
    domain_vec_name: str | None = None


# ----- composite
@dataclasses.dataclass
class DataConfig:
    harmonization: _HarmonizationCfg = field(default_factory=_HarmonizationCfg)
    ingestion: _IngestionCfg = field(default_factory=_IngestionCfg)
    preparation: _PreparationCfg = field(default_factory=_PreparationCfg)
    specification: _Specification = field(default_factory=_Specification)

    def validate(self):
        self.harmonization.validate()
        self.ingestion.validate()
        self.preparation.validate()
