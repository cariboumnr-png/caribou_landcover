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

'''
Taxonomy resolver and gatekeeper validation for ecological domain metadata.

Validates user-declared species taxonomy profiles and code mappings against
canonical knowledge base profiles in `./knowledge`. Resolves raster target
integer classes to deterministic embedding matrix indices.
'''

# standard imports
import difflib
import json
import os
import typing
# local imports
import landseg.geopipe.core as geo_core


# -------------------------------Public Function-------------------------------
def get_available_profiles(
    knowledge_root: str = 'knowledge',
) -> list[str]:
    '''
    Return list of registered taxonomy profile names in knowledge base.

    Args:
        knowledge_root: Root directory of the knowledge base.

    Returns:
        List of profile directory names containing `species_metadata.json`.
    '''
    emb_dir = os.path.join(knowledge_root, 'embeddings')
    if not os.path.isdir(emb_dir):
        return []
    profiles: list[str] = []
    for item in os.listdir(emb_dir):
        sub = os.path.join(emb_dir, item)
        if os.path.isdir(sub):
            meta_path = os.path.join(sub, 'species_metadata.json')
            if os.path.isfile(meta_path):
                profiles.append(item)
    return sorted(profiles)


def resolve_taxonomy_metadata(
    profile: str,
    knowledge_root: str = 'knowledge',
) -> dict[str, typing.Any]:
    '''
    Load canonical metadata JSON for the given taxonomy profile.

    Args:
        profile: Profile name or direct directory path.
        knowledge_root: Root directory of the knowledge base.

    Returns:
        Loaded metadata dictionary from `species_metadata.json`.

    Raises:
        ValueError: If the profile metadata cannot be found.
    '''
    candidate_paths = [
        os.path.join(knowledge_root, 'embeddings', profile, 'species_metadata.json'),
        os.path.join(profile, 'species_metadata.json'),
        profile,
    ]
    meta_path: str | None = None
    for p in candidate_paths:
        if os.path.isfile(p):
            meta_path = os.path.abspath(p)
            break

    if meta_path is None:
        available = get_available_profiles(knowledge_root)
        avail_str = ', '.join(f"'{a}'" for a in available) if available else 'none'
        raise ValueError(
            f"Taxonomy profile '{profile}' not found in '{knowledge_root}'. "
            f"Available profiles: [{avail_str}]."
        )

    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_taxonomy_specs(
    taxonomy: geo_core.TaxonomySpecs,
    num_cls: int,
    knowledge_root: str = 'knowledge',
) -> tuple[geo_core.TaxonomySpecs, dict[str, str], dict[str, int]]:
    '''
    Validate a label layer taxonomy specification against knowledge base.

    Args:
        taxonomy: Taxonomy specification containing profile and mapping.
        num_cls: Number of active classes for the label layer (1..N).
        knowledge_root: Root directory of the knowledge base.

    Returns:
        Tuple containing:
        - normalized `TaxonomySpecs` with canonical indices attached.
        - inferred class names mapping `{'1': 'Black Spruce', ...}`.
        - canonical matrix indices mapping `{'1': 0, ...}`.

    Raises:
        ValueError: On missing profile, invalid mapping, or unknown codes.
    '''
    profile = taxonomy.get('profile')
    if not profile or not isinstance(profile, str):
        raise ValueError(
            "Taxonomy specification must declare a string 'profile' name."
        )

    metadata = resolve_taxonomy_metadata(profile, knowledge_root)
    classes_meta: list[dict[str, typing.Any]] = metadata.get('classes', [])
    if not classes_meta:
        raise ValueError(
            f"Taxonomy metadata for '{profile}' contains no class entries."
        )

    # build lookup maps: exact code, uppercase code, and prefix (e.g. SB)
    code_to_entry: dict[str, dict[str, typing.Any]] = {}
    valid_keys_for_suggest: list[str] = []

    for entry in classes_meta:
        code = entry['code']
        code_to_entry[code] = entry
        code_to_entry[code.upper()] = entry
        valid_keys_for_suggest.append(code)

        # support short FRI code prefix (e.g. 'SB' for 'SB_BLACK_SPRUCE')
        if '_' in code:
            short_prefix = code.split('_', 1)[0]
            if short_prefix not in code_to_entry:
                code_to_entry[short_prefix] = entry
                code_to_entry[short_prefix.upper()] = entry
                valid_keys_for_suggest.append(short_prefix)

    # normalize species mapping
    species_map: dict[str, str] = {}
    if 'species_mapping' in taxonomy and taxonomy['species_mapping']:
        raw_map = taxonomy['species_mapping']
        if not isinstance(raw_map, dict):
            raise ValueError(
                f"Taxonomy 'species_mapping' must be a dictionary, "
                f"got: {type(raw_map)}"
            )
        species_map = {str(k): str(v) for k, v in raw_map.items()}
    elif 'classes' in taxonomy and taxonomy['classes']:
        raw_classes = taxonomy['classes']
        if not isinstance(raw_classes, list):
            raise ValueError(
                f"Taxonomy 'classes' must be a list of string codes, "
                f"got: {type(raw_classes)}"
            )
        species_map = {
            str(i + 1): str(c) for i, c in enumerate(raw_classes)
        }
    else:
        raise ValueError(
            "Taxonomy specification must declare either 'species_mapping' "
            "or 'classes'."
        )

    # verify class count and index boundaries
    expected_indices = {str(i + 1) for i in range(num_cls)}
    actual_indices = set(species_map.keys())
    if actual_indices != expected_indices:
        raise ValueError(
            f"Taxonomy species mapping keys {sorted(actual_indices)} do not "
            f"match expected class indices 1..{num_cls} {sorted(expected_indices)}."
        )

    # resolve entries
    inferred_names: dict[str, str] = {}
    canonical_indices: dict[str, int] = {}
    normalized_mapping: dict[str, str] = {}

    for idx_str in sorted(expected_indices, key=int):
        code_input = species_map[idx_str].strip()
        entry = code_to_entry.get(code_input) or code_to_entry.get(
            code_input.upper()
        )

        if entry is None:
            matches = difflib.get_close_matches(
                code_input, valid_keys_for_suggest, n=3, cutoff=0.4
            )
            sugg_str = (
                f" Did you mean: [{', '.join(f"'{m}'" for m in matches)}]?"
                if matches
                else ''
            )
            raise ValueError(
                f"Unknown species code '{code_input}' for class index {idx_str} "
                f"under profile '{profile}'.{sugg_str}"
            )

        canonical_code = entry['code']
        canonical_idx = entry['index']
        species_name = entry['name']

        normalized_mapping[idx_str] = canonical_code
        canonical_indices[idx_str] = int(canonical_idx)
        inferred_names[idx_str] = species_name

    resolved_taxonomy: geo_core.TaxonomySpecs = {
        'profile': profile,
        'species_mapping': normalized_mapping,
        'canonical_indices': canonical_indices,
    }
    return resolved_taxonomy, inferred_names, canonical_indices
