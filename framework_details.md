# Omega Spatial Control Framework

## What this tool does

`Omega-spatial` is a one-command, dataset-path-driven pipeline for mixed-section spatial transcriptomics data.  
It automates:

1. Dataset detection and loading
2. Schema validation and dataset readiness report
3. QC and preprocessing
4. CNA inference or malignancy scoring (provided or inferred)
5. Pseudo-paired marginal construction within section using malignancy score quantiles
6. Spatially conditioned bridge modeling
7. Counterfactual perturbation inference
8. Gene program discovery (NMF + NMF-vs-PCA/ICA comparison)
9. Benchmarking and validation summaries
10. Figure generation and biologist-facing report output

## Scientific assumptions preserved

- Mixed tumor sections (not externally paired tumor-normal samples)
- Within-section pseudo-paired marginals:
  - `P_tumor = {spots with high CNA score}`
  - `P_normal = {spots with low CNA score}`
- Visium spots treated as mixtures:
  - `x_spot = sum_k w_k x_cell_k`
- Section-level data splitting to reduce leakage

## Command-line usage

```bash
Omega-spatial run --input /path/to/data --output /path/to/results
```

Optional config override:

```bash
Omega-spatial run --input /path/to/data --output /path/to/results --config /path/to/config.yaml
```

Supported inputs include `.h5ad`, `.csv`, `.tsv`, and Visium archives (`.tar`, `.tar.gz`).
If no malignancy/CNA score is provided, the pipeline infers one from expression using chromosome-ordered smoothing and baseline deviation scoring.

## Expected outputs

The pipeline writes:

```text
results/
  annotated_output.h5ad
  perturbation_vectors.csv
  gene_programs.csv
  pathway_enrichment.csv
  benchmark_metrics.csv
  qc_summary.csv
  figures/
  report.html
  report.pdf
  run_config.yaml
  malignancy_scoring_summary.csv
  malignancy_counts_by_section.csv
```

## Folder structure

```text
omega_spatial/
  cli.py                 # Omega-spatial CLI entrypoint
  config.py              # Pipeline config dataclasses and YAML load/dump
  io.py                  # Dataset resolver and loaders
  readiness.py           # Schema validation + dataset readiness report
  cna.py                 # CNA/malignancy scoring inference and marginal assignment
  qc.py                  # Preprocessing and QC
  states.py              # Section split utilities (marginals assigned in cna.py stage)
  spatial.py             # Spatial graph and context aggregation
  model.py               # Conditional score proxy + counterfactual generation
  programs.py            # NMF and NMF-vs-PCA/ICA evaluation
  benchmarks.py          # Baselines and metrics
  reporting.py           # Figures, pathway table, HTML/PDF report generation
  pipeline.py            # End-to-end stage orchestration
```
