# Guide de préparation des données raster

[English](./data_preparation.md) | [Français](./data_preparation_fr.md)

Dernière mise à jour : 2026-08-19

---

## Vue d'ensemble

Les flux de travail d'apprentissage automatique géospatial commencent par une
préparation soignée et cohérente des données. Le cadre `landseg` utilise une
architecture de pipeline déterministe guidée par des artefacts, composée de
quatre étapes séquentielles :

1. **`world-grid`** : Génère et persiste la disposition de tuilage canonique
   (SCR, origine, résolution, dimensions des tuiles et pas de déplacement).
2. **`data-harmonize`** : Rééchantillonne, reprojette et empile les rasters
   bruts multi-sources dans des canevas VRT unifiés alignés sur la grille.
3. **`data-ingest`** : Tuile les rasters harmonisés en blocs `.npz` canoniques
   non partitionnés et matérialise les cartes de domaine.
4. **`data-prepare`** : Partitionne les blocs de données en ensembles
   d'entraînement, validation et test (optionnellement contraints par des
   masques géographiques AOI) et calcule les statistiques de normalisation.

Les utilisateurs peuvent fournir leurs propres fichiers GeoTIFF avec un
manifeste de jeu de données, ou utiliser le script de génération de données
synthétiques pour explorer le cadre.

---

## Contenu

- [Générateur de données d'échantillon synthétiques](#générateur-de-données-déchantillon-synthétiques)
- [Grille mondiale et raster de référence](#grille-mondiale-et-raster-de-référence)
- [Manifeste et configurations de métadonnées](#manifeste-et-configurations-de-métadonnées)
  - [Manifeste central (`manifest.json`)](#manifeste-central-manifestjson)
  - [Rasters de caractéristiques (Optique / MNA)](#rasters-de-caractéristiques-optique--mna)
  - [Rasters d'étiquettes (Têtes de classification)](#rasters-détiquettes-têtes-de-classification)
  - [Rasters de domaine (Contexte optionnel)](#rasters-de-domaine-contexte-optionnel)
  - [Rasters AOI géographiques (Splits optionnels)](#rasters-aoi-géographiques-splits-optionnels)
- [Alignement automatisé vs manuel](#alignement-automatisé-vs-manuel)
- [Tutoriel - Créer des rasters de référence dans QGIS](#tutoriel---créer-des-rasters-de-référence-dans-qgis)

---

## Générateur de données d'échantillon synthétiques

Pour tester l'ensemble du pipeline localement sans données personnalisées, ou
pour inspecter les conventions de nommage de fichiers et les schémas JSON,
générez des rasters d'échantillons synthétiques avec :

```bash
python scripts/generate_dummy_data.py --output_dir ./experiment/input -y
```

Cela génère un jeu de données complet sous `./experiment/input/` :

```text
experiment/input/
├── reference_raster/
│   ├── sample_extent.tif        # Étendue définissant le canevas spatial
│   └── sample_test_aoi.tif      # Masque géographique optionnel de test
└── raw_data/
    ├── manifest.json            # Manifeste central du jeu de données
    ├── sample_dev_sentinel2.tif # Imagerie optique multi-bandes (10m)
    ├── sample_dev_sentinel2.json
    ├── sample_dev_dem.tif       # Raster d'élévation MNA (10m)
    ├── sample_dev_dem.json
    ├── sample_dev_landcover.tif # Vérité terrain couverture du sol
    ├── sample_dev_landcover.json
    ├── sample_dev_leadspc.tif   # Vérité terrain essences forestières
    ├── sample_dev_leadspc.json
    ├── sample_domain_1.tif      # Masque de contexte de domaine
    ├── sample_domain_1.json
    └── ...
```

---

## Grille mondiale et raster de référence

La grille mondiale établit l'ancrage spatial de l'ensemble du projet. Elle doit
être définie dans un système de coordonnées projeté (par ex. `EPSG:3161`) afin
que les coordonnées des tuiles et la taille des pixels correspondent à des
unités métriques linéaires.

La grille est définie via un **raster de référence** (`sample_extent.tif`) :
- Fournit le SCR, la taille de pixel, l'étendue et l'origine haut-gauche.
- Sert de canevas spatial pour l'harmonisation et le découpage en tuiles.

Exécutez le pipeline de grille mondiale :

```bash
landseg pipeline=world-grid
```

Cela génère l'artefact canonique sous :
`experiment/artifacts/world_grids/grid_row_<size>_<stride>_col_<size>_<stride>.json`

---

## Manifeste et configurations de métadonnées

Le pipeline d'harmonisation (`data-harmonize`) lit un fichier central
`manifest.json`. Chaque raster brut référencé dans le manifeste est
accompagné d'un fichier de configuration JSON individuel.

### Manifeste central (`manifest.json`)

Le manifeste est un tableau d'objets associant chaque raster brut à sa
configuration de métadonnées :

```json
[
  {
    "name": "sentinel2",
    "path": "./experiment/input/raw_data/sample_dev_sentinel2.tif",
    "config": "./experiment/input/raw_data/sample_dev_sentinel2.json"
  },
  {
    "name": "dem",
    "path": "./experiment/input/raw_data/sample_dev_dem.tif",
    "config": "./experiment/input/raw_data/sample_dev_dem.json"
  },
  {
    "name": "landcover",
    "path": "./experiment/input/raw_data/sample_dev_landcover.tif",
    "config": "./experiment/input/raw_data/sample_dev_landcover.json"
  },
  {
    "name": "domain_1",
    "path": "./experiment/input/raw_data/sample_domain_1.tif",
    "config": "./experiment/input/raw_data/sample_domain_1.json"
  }
]
```

---

### Rasters de caractéristiques (Optique / MNA)

Les rasters de caractéristiques représentent des variables continues (bandes
satellitaires, élévation, radar). Leur configuration JSON définit un mappage
de bandes indexé à 1 :

**Exemple : Optique multi-bandes (`sample_dev_sentinel2.json`)**
```json
{
  "name": "sentinel2",
  "path": "./experiment/input/raw_data/sample_dev_sentinel2.tif",
  "category": "features",
  "band_mapping": {
    "1": "blue",
    "2": "green",
    "3": "red",
    "4": "red_edge1",
    "5": "red_edge2",
    "6": "red_edge3",
    "7": "nir",
    "8": "narrow_nir",
    "9": "swir1",
    "10": "swir2"
  },
  "label_specs": null
}
```

**Exemple : Modèle numérique d'altitude (`sample_dev_dem.json`)**
```json
{
  "name": "dem",
  "path": "./experiment/input/raw_data/sample_dev_dem.tif",
  "category": "features",
  "band_mapping": {
    "1": "dem"
  },
  "label_specs": null
}
```

---

### Rasters d'étiquettes (Têtes de classification)

Les rasters d'étiquettes contiennent des identifiants de classes entiers pour
la supervision du modèle. Plusieurs rasters d'étiquettes peuvent être fournis
pour entraîner des réseaux multi-tâches.

**Exemple : Couverture du sol (`sample_dev_landcover.json`)**
```json
{
  "name": "landcover",
  "path": "./experiment/input/raw_data/sample_dev_landcover.tif",
  "category": "labels",
  "band_mapping": null,
  "label_specs": {
    "num_cls": 8,
    "ignore_cls": [0, 255]
  }
}
```

- `num_cls` : Nombre total de classes de prédiction actives.
- `ignore_cls` : Liste des valeurs de pixels brutes traitées comme arrière-plan
  ignoré lors du calcul des pertes et de l'évaluation des métriques.

---

### Rasters de domaine (Contexte optionnel)

Les rasters de domaine fournissent des masques contextuels catégoriels (par
exemple districts écologiques, limites administratives) utilisés pour le
conditionnement du modèle.

**Exemple : Contexte de domaine (`sample_domain_1.json`)**
```json
{
  "name": "domain_1",
  "path": "./experiment/input/raw_data/sample_domain_1.tif",
  "category": "domains",
  "band_mapping": null,
  "label_specs": null
}
```

---

### Rasters AOI géographiques (Splits optionnels)

En plus des données du manifeste, les utilisateurs peuvent fournir des rasters
d'aires d'intérêt (AOI) externes pour isoler de façon déterministe des
régions d'évaluation :

- `sample_test_aoi.tif` : Les blocs chevauchant ce raster sont alloués à
  l'ensemble de test (`test`).
- Configuré dans `configs/user.yaml` sous `data-prepare: test_aoi: ...`

---

## Alignement automatisé vs manuel

### Alignement automatisé par pipeline (Recommandé)
Il n'est pas nécessaire d'aligner ou de rééchantillonner manuellement vos
GeoTIFFs bruts au préalable. Le pipeline `data-harmonize` effectue
automatiquement :
1. La reprojection des rasters bruts vers le SCR cible (par ex. `EPSG:3161`).
2. Le rééchantillonnage bilinéaire pour les caractéristiques continues et au
   plus proche voisin pour les étiquettes catégorielles.
3. L'empilement des caractéristiques et étiquettes dans des fichiers `.vrt`.
4. La génération d'un masque de pixels valides (`valid_pixel_mask.vrt`).

Exécutez l'harmonisation :
```bash
landseg pipeline=data-harmonize
```

---

## Tutoriel - Créer des rasters de référence dans QGIS

Pour créer manuellement vos rasters d'étendue de référence ou vos masques AOI :

### Étape 1 — Sélectionner le SCR projeté
1. Ouvrez QGIS.
2. Cliquez sur le bouton SCR en bas à droite.
3. Sélectionnez votre système de coordonnées projeté local (par ex. `EPSG:3161`).

### Étape 2 — Créer la couche d'étendue
1. Allez dans **Couche → Créer une couche → Nouvelle couche de forme**.
2. Définissez le type de géométrie sur **Polygone**.
3. Dessinez un polygone englobant la zone d'étude complète.
4. Enregistrez sous `project_extent.shp`.

### Étape 3 — Rasteuriser l'étendue
1. Allez dans **Raster → Conversion → Rasteurisation (vecteur vers raster)**.
2. Sélectionnez `project_extent.shp` comme entrée.
3. Définissez la valeur fixe à `1`.
4. Définissez la résolution de sortie (par exemple `20` mètres).
5. Enregistrez la sortie sous `sample_extent.tif`.