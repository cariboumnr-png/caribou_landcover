# ADR-0053: Ecological Similarity Loss for Tree Species Segmentation

* **Status:** Accepted and Implemented
* **Date:** 2026-08-31

## 1. Context

The `landseg` training pipeline uses `CompositeLoss` to combine loss functions
such as Focal Loss, Dice Loss, Spectral Smoothness, and Total Variation.

These losses treat segmentation classes as discrete categories. As a result,
all incorrect classes are treated independently of their semantic or
ecological relationship to the target class.

For tree species segmentation, this can be undesirable. For example,
misclassifying Black Spruce (`SB`) as Tamarack (`LA`) may be more ecologically
plausible than misclassifying it as Sugar Maple (`MH`), but a conventional
classification loss does not distinguish these errors.

The repository now contains a knowledge base of tree-species and ecological
group profiles. The profile descriptions are converted into Sentence
Transformer embeddings and a pairwise cosine-similarity matrix is generated
and stored as a reusable artifact.

## 2. Decision

We introducde an **Ecological Similarity Loss** as an optional auxiliary loss
for classification heads that have a compatible taxonomy profile.

### 2.1 Ecological Similarity Loss

For target class $y$, predicted probabilities $p_c$, and similarity matrix
$S$, the ecological loss is:

$$
\mathcal{L}_{eco}
=
\sum_{c=1}^{N} p_c(1-S_{y,c})
$$

The loss therefore gives a smaller penalty to probability assigned to classes
that are more similar to the target and a larger penalty to dissimilar
classes.

The ecological loss is added to the existing composite loss using a configurable
weight:

$$
\mathcal{L}
=
\mathcal{L}_{existing}
+
\lambda_{eco}\mathcal{L}_{eco}
$$

The default ecological weight is `0`, so existing training behavior is
unchanged unless explicitly enabled.

### 2.2 Taxonomy and Knowledge-Base Alignment

A label head can declare a taxonomy profile in its `label_specs`, for example:

```json
{
  "num_cls": 3,
  "taxonomy": {
    "profile": "ontario_tree_species_profiles"
  },
  "class_name": {
    "1": "SB",
    "2": "LA",
    "3": "MH"
  }
}
```

During data harmonization, the declared class codes are validated against the
selected knowledge-base profile. The resolver produces a deterministic
mapping from dataset class indices to the canonical indices in the profile.
Invalid codes or class-count mismatches cause validation to fail.

The resolved taxonomy is carried through the data specification and is used
to associate the appropriate similarity matrix with each prediction head.

The taxonomy mapping is a metadata/indexing contract; it does not by itself
change the raw raster label values.

### 2.3 Similarity Matrix

Similarity matrices are generated offline from the knowledge-base profile
descriptions using a Sentence Transformer. Embeddings are normalized by
default, allowing the pairwise matrix to be computed as the dot product of
the normalized embeddings.

The generated embeddings, similarity matrix, and class metadata are persisted
as knowledge-base artifacts and resolved by profile name when required during
training.

## 3. Consequences

### Positive

* No changes to the UNet architecture or prediction heads.
* No additional trainable model parameters.
* No inference-time computation.
* Ecological relationships can influence training without replacing the
  existing classification losses.
* Taxonomy/profile mismatches are detected during data preparation.
* The similarity matrix is generated once and reused as a versioned artifact.
* The feature is opt-in and backwards compatible through a default weight of
  `0`.

### Negative

* Adds computation during training.
* Introduces an additional hyperparameter, $\lambda_{eco}$.
* Training depends on the quality of the knowledge-base descriptions and the
  embedding model used to construct the similarity matrix.
* The similarity matrix represents an embedding-derived similarity prior; it
  is not a formally validated ecological distance measure.

## Scope

This ADR covers only **Phase 1: ecological similarity loss regularization**.

It does not change the model architecture or introduce ecological features
into the network.

## Future Phases
Future work may consider additional uses of the knowledge base, such as
topographic context conditioning or shared text-visual embeddings. Those
changes should be addressed in separate ADRs:

- **Phase 2 (Topographic Context Conditioning):** Introduce a
  multi-scale spatial neighborhood encoder over DEM, DSM, TPI, and TWI
  channels to condition UNet feature maps via Feature-wise Linear
  Modulation (FiLM).

- **Phase 3 (Shared Text-Visual Embedding Head):** Project UNet
  output features into the Sentence Transformer embedding space to enable
  dot-product zero-shot classification against text vector prompts.
