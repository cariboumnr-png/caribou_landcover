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
Unit tests for pipeline registry (_registry.py).
'''

# third-party imports
import pytest
# local imports
import landseg.execution.pipelines._registry as registry


# ----- `get` helper
@pytest.mark.parametrize('name', [
    'default',
    'data-ingest',
    'data-prepare',
    'diagnose-overfit',
    'model-evaluate',
    'model-train',
    'study-sweep',
    'study-analysis',
])
def test_get_valid_pipeline(name: registry.PipelineName):
    '''
    Given: A valid pipeline name.
    When: `get` is called.
    Then: Return the registered pipeline callable.
    '''
    pipeline_fn = registry.get(name)
    assert callable(pipeline_fn)


def test_get_invalid_pipeline_raises_key_error():
    '''
    Given: An unknown pipeline name.
    When: `get` is called.
    Then: Raise a KeyError.
    '''
    with pytest.raises(KeyError, match='Unknown pipeline name'):
        registry.get('non-existent-pipeline')
