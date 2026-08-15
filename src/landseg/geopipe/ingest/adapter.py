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
    '''Container for harmonized rasters read from the report'''
    domains: dict[str, str] | None
    dev_features: str | None
    dev_labels: str | None
    test_features: str | None
    test_labels: str | None
    valid_mask_raster: str

    def __post_init__(self):
        # here we require dev/test features and labels are in pairs
        if self.dev_features and self.dev_labels is None:
            raise ValueError('Dev features provided but dev labels missing')

        if self.dev_features is None and self.dev_labels:
            raise ValueError('Dev labels provided but dev features missing')

        if self.test_features and self.test_labels is None:
            raise ValueError('Test features provided but test labels missing')

        if self.test_features is None and self.test_labels:
            raise ValueError('Test labels provided but test features missing')

    @property
    def has_dev_data(self) -> bool:
        '''Return `True` if both dev feature and label rasters present.'''
        return self.dev_features is not None and self.dev_labels is not None

    @property
    def has_test_data(self) -> bool:
        '''Return `True` if both test feature and label rasters present.'''
        return self.test_features is not None and self.test_labels is not None


def read_harmonization_report(
    harmonization_paths: artifacts.HarmonizationPaths,
    harmonization_run_id: int | str | None
) -> HarmonizedRasters:
    '''Read Harmonization report to get finalized rasters.'''
    # locate locate targeted/latest harmonization run folder
    try:
        harmonization_paths.get_run_folder(harmonization_run_id)
    except FileNotFoundError as e:
        raise e

    # read report into a typed dict
    try:
        report_path = harmonization_paths.report
        report = ReportController.load_json_or_fail(report_path).fetch()
        assert report
    except artifacts.ArtifactError as e:
        raise e

    finals = report['finalized_rasters']
    assert finals

    # see if domains are present
    domains: dict[str, str] = {}
    for key, value in finals.items():
        if 'domain' in key: # search by tag
            domains.update({key: value})

    return HarmonizedRasters(
        domains=domains,
        dev_features=finals.get('dev_features'),
        dev_labels=finals.get('dev_labels'),
        test_features=finals.get('test_features'),
        test_labels=finals.get('test_labels'),
        valid_mask_raster=harmonization_paths.valid_mask_raster
    )
