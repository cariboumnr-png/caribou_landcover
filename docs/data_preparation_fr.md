# Guide de préparation des données raster

[English](./data_preparation.md) | [Français](./data_preparation_fr.md)

Dernière mise à jour : 2026-08-02

---

## Vue d'ensemble

Des flux de travail efficaces en apprentissage automatique géospatial commencent par une préparation minutieuse et cohérente des données. Avant toute modélisation, entraînement ou expérimentation, tous les rasters d'entrée doivent être standardisés en termes de SCR (CRS), de résolution, d'étendue et d'alignement afin de se comporter de manière prévisible tout au long du pipeline. L'harmonisation de ces propriétés fondamentales dès le départ réduit considérablement la complexité en aval et évite les erreurs liées à des systèmes de référence spatiale incompatibles ou des pixels mal alignés.

Un élément central de cette standardisation est la définition d'une étendue spatiale statique servant de canevas pour le tuilage et l'indexation déterministe. Une fois l'étendue établie, tous les jeux de données destinés à l'entraînement, à l'inférence ou à la production future doivent être calés ou réprojetés pour correspondre à son SCR et à sa taille de pixel. Les utilisateurs peuvent préparer les rasters en externe à l'aide d'outils SIG tels que QGIS, ArcGIS ou GDAL (voir le tutoriel ci-dessous), ou utiliser le pipeline ETL `data-harmonize` intégré au projet (`python scripts/run.py pipeline=data-harmonize`) pour réprojeter et composer automatiquement les GeoTIFF bruts.

Ce guide décrit les étapes recommandées pour créer un raster de référence, appliquer un alignement cohérent sur tous les jeux de données et exporter des rasters propres et compatibles avec la grille, constituant une base solide pour toutes les analyses et modélisations ultérieures.

---

## Sommaire

- [Définition de la grille mondiale](#définition-de-la-grille-mondiale)
- [Spécification des rasters de données d'entrée](#spécification-des-rasters-de-données-dentrée)
  - [Raster d'image](#raster-dimage)
  - [Raster de labels](#raster-de-labels)
  - [Raster de domaine (Optionnel)](#raster-de-domaine-optionnel)
- [Exigences d'alignement des rasters](#exigences-dalignement-des-rasters)
- [JSON de configuration des données](#json-de-configuration-des-données)
- [Disposition de la structure des dossiers du projet](#disposition-de-la-structure-des-dossiers-du-projet)

---

## Définition de la grille mondiale

La grille mondiale doit être définie dans un SCR projeté afin que les coordonnées des tuiles et les dimensions des pixels correspondent à des unités linéaires (par exemple, des mètres). Le point de départ consiste à établir **l'étendue du projet**—la zone complète englobant à la fois la région d'entraînement et toutes les régions de prédiction prévues. Cette étendue doit être considérée comme **immutable pour l'ensemble du projet**, garantissant que toutes les étapes ultérieures de préparation des données fassent référence au même domaine spatial.

Une fois l'étendue fixée, les utilisateurs peuvent générer une ou plusieurs **grilles mondiales** en tant qu'artefacts stables et versionnés pour différents besoins expérimentaux. Par exemple, les grilles peuvent varier selon la taille des tuiles (affectant le champ de vision du modèle) ou inclure/omettre un chevauchement de tuiles pour étudier les effets de bord. Bien que les grilles puissent changer d'une expérience à l'autre, elles doivent toutes rester ancrées au même SCR, à la même résolution et à la même origine définis par l'étendue du projet. Ce projet suppose que les rasters sont toujours ancrés **en haut à gauche**.

L'étendue de la grille peut être fournie de deux manières :

  - Définition manuelle à l'aide d'une origine en haut à gauche et d'un nombre spécifié de tuiles dans les directions horizontale et verticale.
  - Définition par raster de référence (préférée), où un raster créé dans des outils SIG courants (QGIS, ArcGIS, GDAL) fournit le SCR, la résolution en pixels, l'étendue et l'origine pour construire la grille.

Après la définition de l'étendue du projet, les grilles mondiales en sont dérivées au cours du pipeline (module `landseg.geopipe.ingest.world_grids`) pour former des schémas de tuilage reproductibles et versionnés utilisés tout au long de l'expérimentation et de la production.

[Sauter](#tutoriel---créer un raster de référence) au tutoriel sur la création d'un raster de référence dans QGIS.

<img src="./images/extent_reference.png" alt="extent_reference" width="800">

**Figure 1**. Création du raster de référence d'étendue.

---

## Spécification des rasters de données d'entrée

### Raster d'image
Les rasters d'images utilisés pour l'entraînement et la prédiction du modèle proviennent généralement de plateformes satellitaires telles que *Landsat*, accessibles soit par le [portail USGS EarthExplorer](https://earthexplorer.usgs.gov/), soit par [Google Earth Engine (GEE)](https://earthengine.google.com/). Vous pouvez choisir le flux de travail avec lequel vous êtes le plus à l'aise.

>**Remarque :** la sélection des scènes, le mosaïquage, le masquage des nuages et les autres décisions QA/QC restent en dehors du périmètre de ce framework, car ils dépendent fortement des exigences spécifiques au projet et de l'expertise de l'utilisateur.

Pour les utilisateurs de GEE, nous recommandons d'explorer le flux de travail **Best Available Pixel (BAP)**, qui fournit des outils flexibles pour assembler des composites annuels de haute qualité. Une mise en œuvre largement adoptée est disponible ici : <https://github.com/saveriofrancini/bap>. Le compositage de type BAP permet de produire des rasters temporellement stables et sans nuages adaptés aux modèles ML en aval.

Le composite d'image d'entrée est flexible : les utilisateurs peuvent fournir n'importe quel ensemble de canaux raster, sans exigence rigide quant au nombre ou à l'ordre des bandes. Les caractéristiques dérivées optionnelles—telles que les indices spectraux (par ex. NDVI, NDWI) ou les couches topographiques issues d'un MNE (par ex. pente, aspect)—sont calculées automatiquement par les pipelines en aval *uniquement si* les bandes optiques ou la couche d'élévation requises sont répertoriées dans `image_band_map`. Les utilisateurs peuvent fournir autant ou aussi peu de canaux que leur application l'exige.

<img src="./images/example_image_raster.png" alt="example_image_raster" width="800">

**Figure 2**. Exemple de raster d'image.

---

### Raster de labels
Les rasters de labels sont entièrement définis par l'utilisateur, car le système d'étiquetage provient des connaissances du domaine de l'utilisateur, de ses sources de données et des objectifs du projet. Ce framework ne prescrit aucun schéma de classification spécifique ; il s'attend plutôt à ce que les utilisateurs fournissent un raster contenant les labels de couverture du sol ou de segmentation pertinents pour leur flux de travail.

Comme le projet est conçu pour la segmentation de la couverture du sol, le raster de labels doit contenir :

  - Des identifiants de classe `Entier`, représentant les catégories de couverture du sol.
  - Une valeur `NoData` clairement définie.
  - Toutes les classes que l'utilisateur a l'intention d'ignorer pendant l'entraînement (par exemple, l'eau, les nuages, les zones non classées).

Pendant la préparation des données, `NoData` et les classes à ignorer spécifiées par l'utilisateur sont automatiquement convertis en un seul indice de label ignoré (généralement 255, configurable par l'utilisateur). Cela garantit un traitement propre des pixels invalides ou indésirables tout au long de l'entraînement et de l'inférence.

Dans de nombreux systèmes de classification réels, le nombre de classes de couverture du sol brutes peut être élevé, déséquilibré ou difficile à modéliser efficacement en une seule passe. Pour soutenir des stratégies d'entraînement plus gérables et étagées, ce framework fournit une hiérarchie de labels parent–enfant optionnelle à deux niveaux :

  - Les classes parents représentent des groupes généralisés plus larges.
  - Les classes enfants représentent les catégories brutes à plus fine échelle appartenant à chaque groupe parent.

Cette hiérarchie permet des flux de travail tels que :

  1. L'entraînement d'un modèle initial sur des groupes parents grossiers pour apprendre la structure globale.
  2. L'affinage du modèle en se concentrant sur certains groupes parents sélectionnés et en s'entraînant sur les classes enfants complètes qui leur sont associées.

Si vous souhaitez utiliser cette approche hiérarchique, vous devez fournir une configuration JSON qui définit les correspondances parent–enfant. Le format et l'utilisation de cette configuration sont décrits plus loin dans le guide.

<img src="./images/example_label_raster.png" alt="example_label_raster" width="800">

**Figure 3**. Exemple de raster de labels.

---

### Raster de domaine (Optionnel)
Un raster de domaine est un jeu de données ***optionnel*** qui peut être inclus lorsque l'étude bénéficie de la spécification de sous-régions écologiques, géographiques ou de gestion. Le domaine peut représenter n'importe quel découpage défini par l'utilisateur et pertinent pour le projet—écozones, limites administratives, régimes de perturbation, strates biophysiques ou autres divisions contextuelles. Bien qu'optionnel pour l'entraînement, un raster de domaine devrait idéalement **couvrir à la fois la région d'entraînement et la zone de prédiction prévue** pour assurer un conditionnement cohérent sur l'ensemble de l'étendue du projet.

Le raster de domaine doit être à **valeurs entières**, chaque entier représentant une catégorie de domaine unique. Les utilisateurs n'ont pas besoin de prétraiter ces valeurs au-delà de s'assurer de leur exactitude ; pendant l'entraînement, le framework convertit automatiquement le raster de domaine brut dans les représentations internes requises par la stratégie de conditionnement choisie.

Comme le traitement du domaine a lieu dans la configuration d'entraînement—et non lors de l'étape de préparation des données—ce guide exige uniquement que les utilisateurs fournissent un raster de domaine propre, encodé en entiers et aligné sur l'étendue du projet et le raster de référence.

<img src="./images/example_domain_raster.png" alt="example_domain_raster" width="800">

**Figure 4**. Exemple de raster de domaine.

---

## Exigences d'alignement des rasters

Tous les rasters d'entrée—image, label et domaine optionnel—doivent être **alignés** sur le raster de référence du projet créé lors de la définition de la grille mondiale. Cela garantit que chaque raster partage :

- **Le même SCR projeté**
- **La même résolution en pixels**
- **La même origine et le même alignement de pixels**

Le calage sur le raster de référence garantit que les limites des pixels correspondent exactement, ce qui est essentiel pour un tuilage déterministe, un appariement correct label-image et des expériences reproductibles.

Tous les rasters doivent également se trouver **entièrement dans les limites** de l'étendue du projet. Toutes les données s'étendant au-delà de l'étendue du raster de référence seront découpées ou éliminées lors de l'alignement.

[Sauter](#tutoriel---flux de travail d'alignement dans qgis) au tutoriel sur l'alignement des rasters sur un raster de référence dans QGIS.

---

## JSON de configuration des données

Un fichier JSON de configuration des données accompagne les rasters d'entrée. Il définit l'ordre des bandes, les spécifications des labels et le remappage optionnel des classes.

### Champs de configuration principaux
| Key | Purpose (FR) | Notes |
|-----|--------------|-------|
| `image_band_map` | Définit l'ordre des canaux du composite d'image | Doit être une correspondance d'indexation commençant à 0 |
| `label_specs` | Définit les spécifications de labels de chaque tâche | Contient `num_cls`, `ignore_cls` et `reclass_map` |

**Exemple :**
```json
{
  "image_band_map": {
    "dem": 0, "blue": 1, "green": 2,
    "red": 3, "nir": 4, "swir1": 5, "swir2": 6
  },
  "label_specs": {
    "main_task": {
      "num_cls": 8,
      "ignore_cls": [0, 255],
      "reclass_map": {
        "1": 1, "2": 1,
        "3": 2, "4": 2,
        "5": 3, "6": 3
      }
    }
  }
}
```

### Champs de métadonnées optionnels
Ces éléments améliorent l'interprétabilité et la visualisation, mais ne sont pas requis par le pipeline de prétraitement.

| Key | Purpose (FR) |
|-----|--------------|
| `label_class_name` | Noms lisibles par l'humain pour les catégories de labels bruts |
| `label_reclass_name` | Noms lisibles par l'humain pour les classes parents |
| `label_reclass_color_map` | Tableaux de couleurs RVB pour l'aperçu visuel des classes |

**Règles clés**
- Les indices de bandes dans `image_band_map` doivent commencer à 0.
- Les valeurs de `ignore_cls` (telles que `0` ou `255`) sont automatiquement gérées lors de l'ingestion des données.
- `reclass_map` est optionnel ; à utiliser uniquement si vous exploitez un regroupement de classes parent-enfant.

---

## Disposition de la structure des dossiers du projet

Les rasters d'entrée, les artefacts de pipeline générés et les résultats d'exécution de session sont organisés au sein d'un répertoire racine d'expérience structuré (`<exp_root>`).

Pour l'arborescence complète et détaillée des dossiers d'entrée, d'artefacts et de résultats d'entraînement, consultez :
- [Structure et Arborescence du Répertoire des Expériences et Artefacts](./experiment_directory_layout_fr.md) ([English](./experiment_directory_layout.md))

> **Génération de données factices** : Pour alimenter un environnement de test local avec un ensemble complet de GeoTIFFs synthétiques, exécutez :
> ```bash
> python scripts/generate_dummy_data.py
> ```

---

## Tutoriel - Créer un raster de référence

### Étape 1 — Sélectionner le SCR projeté
1. Ouvrez QGIS.
2. Dans le coin inférieur droit, cliquez sur le bouton SCR.
3. Sélectionnez votre système projeté local (par exemple, `EPSG:3161`).

---

### Étape 2 — Créer la couche d'étendue
1. Allez dans **Couche → Créer une couche → Nouvelle couche de fichier de forme (Shapefile)**.
2. Définissez le type de géométrie sur **Polygone**.
3. Dessinez un polygone englobant qui couvre entièrement :
   - Toutes les imageries d'entraînement
   - Toutes les régions de prédiction prévues
4. Enregistrez le fichier sous le nom `project_extent.shp`.

---

### Étape 3 — Rasteriser l'étendue
1. Allez dans **Raster → Conversion → Rasteriser (Vecteur vers Raster)**.
2. Sélectionnez `project_extent.shp` comme entrée.
3. Définissez la valeur fixe sur `1`.
4. Choisissez la résolution cible (par exemple, `20` mètres).
5. Format de sortie : **GeoTIFF**.
6. Enregistrez sous le nom `reference_extent.tif`.

---

### Tutoriel - Flux de travail d'alignement dans QGIS

**Tâche 1 — Charger les données**<br>
1. Ouvrez QGIS.
2. Glissez-déposez :
   - Le **raster d'étendue de référence**.
   - Votre **raster d'image**.
   - Votre **raster de labels**.
   - Votre **raster de domaine** (optionnel).

---

**Tâche 2 — Ouvrir l'outil d'alignement**<br>
1. Ouvrez la **Boîte d'outils de traitement**.
2. Naviguez vers : **GDAL → Alignement raster → Aligner les rasters**.

---

**Tâche 3 — Configurer les paramètres d'alignement**<br>
1. **Couche d'entrée :** Sélectionnez le raster à aligner.
2. **Couche de référence :** Choisissez le **raster d'étendue de référence**.
3. **Taille du raster de sortie :**
   - Résolution cible : **Résolution de la couche**
   - SCR cible : automatiquement extrait du raster de référence
4. **Alignement de sortie :**
   - Activez **Faire correspondre l'alignement des pixels**
   - Activez **Découper selon l'étendue de la couche de référence**

---

**Tâche 4 — Enregistrer le raster aligné**<br>
Enregistrez sous `image_aligned.tif`, `labels_aligned.tif` et `domain_aligned.tif`.