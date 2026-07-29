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
Unit tests for `landseg.configs.schema.sections.dataspecs`.
'''

# local imports
import landseg.configs.schema.sections.dataspecs as dataspecs


# ----- `DataSpecs` tests
def test_data_specs_default_instantiation() -> None:
    '''verify `DataSpecs` default attribute values.'''
    specs = dataspecs.DataSpecs()
    assert specs.domain_ids_name is None
    assert specs.domain_vec_name is None


def test_data_specs_custom_values() -> None:
    '''verify `DataSpecs` assignment with custom domain names.'''
    specs = dataspecs.DataSpecs(
        domain_ids_name='eco_region',
        domain_vec_name='climate_vec',
    )
    assert specs.domain_ids_name == 'eco_region'
    assert specs.domain_vec_name == 'climate_vec'
