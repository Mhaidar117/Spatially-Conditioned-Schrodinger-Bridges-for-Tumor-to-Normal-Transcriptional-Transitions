# Spatial Perturbations: Optimal Control for Tumor-Normal Tissue Transitions

A computational framework for learning minimal transcriptional perturbations that transform tumor tissue states toward healthy reference states, using spatially coherent optimal control and Schrödinger bridges.

## Motivation

Standard differential expression analysis identifies what differs between tumor and normal tissue but does not address a deeper question: what minimal transcriptional perturbations would suffice to transform one tissue state into another? Conventional approaches treat genes independently and discard spatial context, missing the coordinated, spatially organized nature of tissue-level transformation. In glioblastoma (GBM), spatial transcriptomics reveals continuous gradients from the hypoxic tumor core through infiltrating margins to adjacent normal brain. This graded spatial architecture motivates a control-theoretic approach that reasons about interventions rather than associations.

## Goal

This project formulates tumor-normal transitions as a stochastic optimal control problem. The objective is to learn a spatially coherent control law (a *rejuvenation vector field*) that transports cells from a perturbed state (tumor) to a reference state (healthy) with minimal energy cost, using the framework of Schrödinger bridges (entropy-regularized optimal transport).

## Methods

The tumor-to-normal transformation is modeled as a Schrödinger bridge between the two observed marginal distributions. The bridge recovers the most likely stochastic process connecting these marginals under a maximum-entropy constraint, providing a principled operationalization of "minimal perturbation." The bridge drift is parameterized via a conditional score-based diffusion model conditioned on spatial context derived from each spot's local tissue neighborhood via graph-based aggregation. Conditioning on spatial context ensures that neighboring spots with similar microenvironments receive similar drift estimates, producing spatially coherent trajectories. The difference between the transported (counterfactual) state and the observed tumor state at each spot defines the minimal transcriptional perturbation required to restore the healthy state. The resulting spatially resolved perturbation matrix is decomposed into interpretable gene programs via NMF and mapped back to tissue coordinates.

## Data

The framework uses the Greenwald et al. (2024) GBM Visium dataset (~70,000 spots across 19 glioma sections), which provides multi-region sampling (necrotic core, contrast-enhancing, infiltrating edge), continuous copy number aberration-based malignancy scores per spot, and a well-characterized 5-layer spatial organization (L1-L5). Source and target distributions are defined using the CNA malignancy gradient. The Ravi et al. (2022) dataset (13 additional GBM Visium sections) serves as a held-out validation cohort.

## Project Structure

- `omega_spatial/` - Core Python package implementing the spatial control pipeline
- `data_exploration.ipynb` - Visium spatial transcriptomics exploration and preprocessing
- `omega_spatial.default.yaml` - Default configuration
- `Docs/` - Project documentation and background literature

## Installation

Requires Python 3.10 or later. Install in development mode:

```bash
pip install -e .
```

## Usage

The pipeline can be run via the command-line interface:

```bash
Omega-spatial run --config omega_spatial.default.yaml
```

## Expected Deliverables

1. A Python tool that, given paired spatial transcriptomics h5ad files, estimates spatially coherent minimal perturbation vectors via the conditioned Schrödinger bridge and outputs an annotated h5ad with per-spot rejuvenation vectors and gene program scores
2. A synthetic spatial simulation framework with known perturbation structure for method validation
3. A benchmark comparing inferred control signals to differential expression and static optimal transport baselines (e.g., CellOT)
4. Documentation of methodology and results

## References

- De Bortoli et al. (2021). Diffusion Schrödinger bridge with applications to score-based generative modeling. NeurIPS.
- Greenwald et al. (2024). Integrative spatial analysis reveals a multi-layered organization of glioblastoma. Cell.
- Bunne et al. (2023). Learning single-cell perturbation responses using neural optimal transport. Nature Methods.
- Ravi et al. (2022). Spatially resolved multi-omics deciphers bidirectional tumor-host interdependence in glioblastoma. Cancer Cell.

## License

See LICENSE file for details.
