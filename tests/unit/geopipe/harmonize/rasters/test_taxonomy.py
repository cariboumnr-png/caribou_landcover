# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''Unit tests for taxonomy resolver and gatekeeper validation.'''

# standard imports
import json
# third-party imports
import pytest
# local imports
import landseg.geopipe.harmonize.rasters.taxonomy as taxonomy


# ----- `get_available_profiles` tests
def test_get_available_profiles_finds_registered_profiles():
    '''
    Given: The canonical knowledge base directory.
    When: `get_available_profiles` is called.
    Then: Return list containing registered profiles.
    '''
    profiles = taxonomy.get_available_profiles('knowledge')
    assert 'ontario_tree_species_grouped_profiles' in profiles
    assert 'ontario_tree_species_profiles' in profiles


def test_get_available_profiles_nonexistent_dir(tmp_path):
    '''
    Given: A non-existent root path.
    When: `get_available_profiles` is called.
    Then: Return an empty list without raising.
    '''
    empty_root = str(tmp_path / 'nonexistent')
    profiles = taxonomy.get_available_profiles(empty_root)
    assert profiles == []


# ----- `resolve_taxonomy_metadata` tests
def test_resolve_taxonomy_metadata_success():
    '''
    Given: A registered profile name.
    When: `resolve_taxonomy_metadata` is called.
    Then: Return parsed dictionary with classes and model info.
    '''
    meta = taxonomy.resolve_taxonomy_metadata(
        'ontario_tree_species_grouped_profiles',
        knowledge_root='knowledge',
    )
    assert 'classes' in meta
    assert len(meta['classes']) > 0
    assert meta['classes'][0]['code'] == 'SB_BLACK_SPRUCE'


def test_resolve_taxonomy_metadata_not_found(tmp_path):
    '''
    Given: An invalid profile name.
    When: `resolve_taxonomy_metadata` is called.
    Then: Raise ValueError detailing available profiles.
    '''
    with pytest.raises(ValueError, match="Taxonomy profile 'invalid_prof'"):
        taxonomy.resolve_taxonomy_metadata(
            'invalid_prof',
            knowledge_root=str(tmp_path),
        )


# ----- `validate_taxonomy_specs` tests
def test_validate_taxonomy_specs_species_mapping():
    '''
    Given: A valid taxonomy spec with explicit species_mapping.
    When: `validate_taxonomy_specs` is called.
    Then: Return normalized taxonomy, class names, and matrix indices.
    '''
    spec = {
        'profile': 'ontario_tree_species_grouped_profiles',
        'species_mapping': {
            '1': 'SB_BLACK_SPRUCE',
            '2': 'PJ_JACK_PINE',
        },
    }
    resolved, names, indices = taxonomy.validate_taxonomy_specs(
        spec,
        num_cls=2,
        knowledge_root='knowledge',
    )
    assert resolved['profile'] == 'ontario_tree_species_grouped_profiles'
    assert resolved['species_mapping']['1'] == 'SB_BLACK_SPRUCE'
    assert resolved['species_mapping']['2'] == 'PJ_JACK_PINE'
    assert indices['1'] == 0
    assert indices['2'] == 3
    assert names['1'] == 'Black Spruce'
    assert names['2'] == 'Jack Pine'


def test_validate_taxonomy_specs_short_fri_prefix():
    '''
    Given: A taxonomy spec using short FRI code prefixes (SB, PJ).
    When: `validate_taxonomy_specs` is called.
    Then: Canonicalize codes to full profile entries.
    '''
    spec = {
        'profile': 'ontario_tree_species_grouped_profiles',
        'species_mapping': {
            '1': 'SB',
            '2': 'PJ',
        },
    }
    resolved, names, indices = taxonomy.validate_taxonomy_specs(
        spec,
        num_cls=2,
        knowledge_root='knowledge',
    )
    assert resolved['species_mapping']['1'] == 'SB_BLACK_SPRUCE'
    assert resolved['species_mapping']['2'] == 'PJ_JACK_PINE'
    assert indices['1'] == 0
    assert indices['2'] == 3


def test_validate_taxonomy_specs_classes_list():
    '''
    Given: A taxonomy spec using shorthand classes list.
    When: `validate_taxonomy_specs` is called.
    Then: Auto-convert list to 1-based species mapping.
    '''
    spec = {
        'profile': 'ontario_tree_species_grouped_profiles',
        'classes': ['SB_BLACK_SPRUCE', 'SW_WHITE_SPRUCE'],
    }
    resolved, names, indices = taxonomy.validate_taxonomy_specs(
        spec,
        num_cls=2,
        knowledge_root='knowledge',
    )
    assert resolved['species_mapping']['1'] == 'SB_BLACK_SPRUCE'
    assert resolved['species_mapping']['2'] == 'SW_WHITE_SPRUCE'
    assert indices['1'] == 0
    assert indices['2'] == 1


def test_validate_taxonomy_specs_unknown_code_suggestion():
    '''
    Given: A species code with a typo (e.g. 'SBB').
    When: `validate_taxonomy_specs` is called.
    Then: Raise ValueError with candidate suggestions.
    '''
    spec = {
        'profile': 'ontario_tree_species_grouped_profiles',
        'species_mapping': {
            '1': 'SBB',
            '2': 'PJ_JACK_PINE',
        },
    }
    with pytest.raises(ValueError, match="Unknown species code 'SBB'"):
        taxonomy.validate_taxonomy_specs(
            spec,
            num_cls=2,
            knowledge_root='knowledge',
        )


def test_validate_taxonomy_specs_index_mismatch():
    '''
    Given: Species mapping keys that do not match 1..num_cls.
    When: `validate_taxonomy_specs` is called.
    Then: Raise ValueError with index mismatch explanation.
    '''
    spec = {
        'profile': 'ontario_tree_species_grouped_profiles',
        'species_mapping': {
            '1': 'SB_BLACK_SPRUCE',
            '3': 'PJ_JACK_PINE',
        },
    }
    with pytest.raises(ValueError, match='do not match expected class indices'):
        taxonomy.validate_taxonomy_specs(
            spec,
            num_cls=2,
            knowledge_root='knowledge',
        )
