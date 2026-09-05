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
Resolution utilities for feature channels and target reclassification.

Parses user-specified preparation configurations (named schemes or
inline overrides) against ingested catalog schemas to resolve active
input feature channels and multi-head target hierarchies.
'''

# standard imports
import typing


# ------------------------------private constants------------------------------
_ENGINEERED_GROUPS: dict[str, tuple[tuple[str, ...], str]] = {
    'topo': (
        ('slope', 'cos_aspect', 'sin_aspect', 'tpi'),
        'add_topo: true',
    ),
    'topography': (
        ('slope', 'cos_aspect', 'sin_aspect', 'tpi'),
        'add_topo: true',
    ),
    'spectral': (
        ('ndvi', 'ndmi', 'nbr'),
        'add_spectral',
    ),
    'spectral_indices': (
        ('ndvi', 'ndmi', 'nbr'),
        'add_spectral',
    ),
    'indices': (
        ('ndvi', 'ndmi', 'nbr'),
        'add_spectral',
    ),
}


# ------------------------------private functions------------------------------
def _resolve_engineered_group(
    group_name: str,
    selection: typing.Any,
    available_band_map: typing.Mapping[str, int],
) -> list[str]:
    '''
    Resolve bands for engineered pseudo-datasets (topo or spectral).
    '''
    known_bands, ingest_hint = _ENGINEERED_GROUPS[group_name]
    if selection is False:
        return []

    is_all = (
        selection is None
        or selection is True
        or (
            isinstance(selection, str)
            and selection.lower() in (
                'all',
                'use topo layers',
                'use spectral indices',
            )
        )
    )
    if is_all:
        matched = [b for b in available_band_map if b in known_bands]
        if not matched:
            raise ValueError(
                f'Engineered features requested for "{group_name}", but '
                f'no matching bands ({list(known_bands)}) exist in data '
                f'blocks. Ensure "{ingest_hint}" was set during data '
                'ingestion.'
            )
        return matched

    if isinstance(selection, str):
        band = selection.lower()
        if band in available_band_map and band in known_bands:
            return [band]
        available_known = [
            b for b in available_band_map if b in known_bands
        ]
        raise ValueError(
            f'Band "{selection}" for "{group_name}" not found in data '
            f'blocks. Available: {available_known}'
        )

    if isinstance(selection, list):
        result: list[str] = []
        for item in selection:
            band = str(item).lower()
            if band not in available_band_map:
                raise ValueError(
                    f'Band "{band}" for "{group_name}" not found in data '
                    f'blocks. Ensure "{ingest_hint}" was set during '
                    'data ingestion.'
                )
            result.append(band)
        return result

    return []


# ----- public functions
def resolve_feature_channels(
    available_band_map: typing.Mapping[str, int],
    user_features_cfg: (
        typing.Mapping[str, str | list[str] | bool] | None
    ) = None,
    raster_schemes: (
        typing.Mapping[str, typing.Mapping[str, list[str]]] | None
    ) = None,
) -> tuple[list[str], list[int]]:
    '''
    Resolve active feature band names and 0-based channel indices.

    If `user_features_cfg` is empty or None, all available bands in
    `available_band_map` are selected in sequential order.

    Args:
        available_band_map: Mapping of lower-case band names to 0-based
            channel indices in the ingested data blocks.
        user_features_cfg: Mapping of raster dataset name to scheme name
            (e.g. 'rgb_nir', 'all'), inline list of band names, or
            engineered pseudo-dataset toggles ('topo', 'spectral').
        raster_schemes: Mapping of raster dataset name to its named
            schemes dictionary from dataset manifest metadata.

    Returns:
        A tuple of (selected_band_names, selected_channel_indices).
    '''
    if not user_features_cfg:
        names = sorted(
            available_band_map.keys(), key=lambda k: available_band_map[k]
        )
        return names, [available_band_map[b] for b in names]

    selected_names: list[str] = []
    schemes_dict = raster_schemes or {}

    for raster_name, selection in user_features_cfg.items():
        r_lower = raster_name.lower()
        if r_lower in _ENGINEERED_GROUPS:
            selected_names.extend(
                _resolve_engineered_group(
                    r_lower, selection, available_band_map
                )
            )
            continue

        if selection is None or selection == 'all':
            # select all bands that contain raster prefix or match
            matched = [
                b for b in available_band_map
                if b == raster_name or b.startswith(f'{raster_name}_')
            ]
            if not matched:
                matched = [b for b in available_band_map if b == raster_name]
            selected_names.extend(matched)
            continue

        if isinstance(selection, str):
            # look up named scheme in raster schemes
            r_schemes = schemes_dict.get(raster_name, {})
            if not r_schemes and selection in schemes_dict:
                r_schemes = schemes_dict
            if selection not in r_schemes:
                raise ValueError(
                    f'Named feature scheme "{selection}" not found for '
                    f'raster "{raster_name}". Available: '
                    f'{list(r_schemes.keys())}'
                )
            selected_names.extend(r_schemes[selection])

        elif isinstance(selection, list):
            # inline list of band names
            for band in selection:
                if band not in available_band_map:
                    raise ValueError(
                        f'Band "{band}" in feature selection not found in '
                        f'available bands: {list(available_band_map.keys())}'
                    )
                selected_names.append(band)

    # deduplicate while preserving order
    deduped_names = [
        b for b in dict.fromkeys(selected_names) if b in available_band_map
    ]

    if not deduped_names:
        # fallback to all available if selection matched nothing
        deduped_names = sorted(
            available_band_map.keys(), key=lambda k: available_band_map[k]
        )

    selected_indices = [available_band_map[b] for b in deduped_names]
    return deduped_names, selected_indices


def resolve_target_reclass(
    label_names_map: typing.Mapping[str, list[str]] | typing.Sequence[str],
    user_targets_cfg: (
        typing.Mapping[str, str | dict[str, typing.Any]] | None
    ) = None,
    raster_schemes: (
        typing.Mapping[str, typing.Mapping[str, typing.Any]] | None
    ) = None,
) -> dict[str, dict[str, typing.Any] | None]:
    '''
    Resolve active reclassification settings per target label layer.

    Args:
        label_names_map: Mapping of label layer name to list of class
            names or sequence of label layer names.
        user_targets_cfg: Mapping of label name to scheme name or inline
            reclassification dictionary.
        raster_schemes: Mapping of raster name to named label schemes.

    Returns:
        Mapping of label layer name to resolved LabelScheme or None.
    '''
    resolved: dict[str, dict[str, typing.Any] | None] = {}
    if not user_targets_cfg:
        return {k: None for k in label_names_map}

    schemes_dict = raster_schemes or {}

    for label_name in label_names_map:
        cfg = user_targets_cfg.get(label_name)
        if cfg is None or cfg in ('raw', 'base', 'none'):
            resolved[label_name] = None
            continue

        if isinstance(cfg, str):
            r_schemes = schemes_dict.get(label_name, {})
            if not r_schemes and cfg in schemes_dict:
                r_schemes = schemes_dict
            if cfg not in r_schemes:
                raise ValueError(
                    f'Named target scheme "{cfg}" not found for label '
                    f'"{label_name}". Available: {list(r_schemes.keys())}'
                )
            resolved[label_name] = dict(r_schemes[cfg])

        elif isinstance(cfg, dict):
            resolved[label_name] = cfg

    return resolved
