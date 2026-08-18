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

'''
Adapter for harmonization -> ingestion.
'''

# standard imports
import dataclasses
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.harmonize as harmonize

# aliases
ReportController = artifacts.Controller[harmonize.HarmonizationReportSchema]


@dataclasses.dataclass
class HarmonizedRasters:
    '''Container for harmonized rasters read from the report.'''
    domains: dict[str, str] | None
    features: str | None
    labels: str | None
    valid_mask_raster: str
    world_grid_fpath: str | None = None
    dev_features: str | None = None
    dev_labels: str | None = None
    test_features: str | None = None
    test_labels: str | None = None

    def __post_init__(self):
        if self.features is None and self.dev_features is not None:
            self.features = self.dev_features
        if self.labels is None and self.dev_labels is not None:
            self.labels = self.dev_labels
        if self.dev_features is None and self.features is not None:
            self.dev_features = self.features
        if self.dev_labels is None and self.labels is not None:
            self.dev_labels = self.labels

    @property
    def has_data(self) -> bool:
        '''Return True if both feature and label rasters are present.'''
        return self.features is not None and self.labels is not None

    @property
    def has_dev_data(self) -> bool:
        '''Backward compatibility alias for has_data.'''
        return self.has_data

    @property
    def has_test_data(self) -> bool:
        '''Backward compatibility property.'''
        return self.test_features is not None and self.test_labels is not None


def read_harmonization_report(
    harmonization_paths: artifacts.HarmonizationPaths,
    harmonization_run_id: int | str | None
) -> HarmonizedRasters:
    '''Read Harmonization report to get finalized rasters.'''
    # locate targeted/latest harmonization run folder
    harmonization_paths.get_run_folder(harmonization_run_id)

    # read report into a typed dict
    report_path = harmonization_paths.report
    report = ReportController.load_json_or_fail(report_path).fetch()
    assert report

    finals = report['finalized_rasters']
    assert finals

    # see if domains are present
    domains: dict[str, str] = {}
    for key, value in finals.items():
        if 'domain' in key: # search by tag
            domains.update({key: value})

    features = finals.get('features') or finals.get('dev_features')
    labels = finals.get('labels') or finals.get('dev_labels')

    world_grid_fpath = None
    if report.get('world_grid'):
        world_grid_fpath = report['world_grid'].get('grid_fpath')

    return HarmonizedRasters(
        domains=domains,
        features=features,
        labels=labels,
        valid_mask_raster=harmonization_paths.valid_mask_raster,
        world_grid_fpath=world_grid_fpath,
        dev_features=finals.get('dev_features'),
        dev_labels=finals.get('dev_labels'),
        test_features=finals.get('test_features'),
        test_labels=finals.get('test_labels'),
    )
