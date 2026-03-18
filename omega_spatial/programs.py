from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, NMF, PCA

from .config import PipelineConfig


@dataclass
class ProgramResults:
    scores: pd.DataFrame
    loadings: pd.DataFrame
    model_selection: pd.DataFrame
    chosen_method: str


def _nonnegative(a: np.ndarray) -> np.ndarray:
    return np.clip(a, a_min=0.0, a_max=None)


def compare_factorizations(perturb: np.ndarray, cfg: PipelineConfig) -> pd.DataFrame:
    n = max(2, min(cfg.programs.chosen_components, perturb.shape[1] - 1))
    x = perturb
    x_nonneg = _nonnegative(x - x.min())

    nmf = NMF(n_components=n, init="nndsvda", random_state=cfg.programs.random_seed, max_iter=500)
    w_nmf = nmf.fit_transform(x_nonneg)
    h_nmf = nmf.components_
    rec_nmf = np.linalg.norm(x_nonneg - w_nmf @ h_nmf) / np.linalg.norm(x_nonneg)

    pca = PCA(n_components=n, random_state=cfg.programs.random_seed)
    w_pca = pca.fit_transform(x)
    h_pca = pca.components_
    rec_pca = np.linalg.norm(x - w_pca @ h_pca) / np.linalg.norm(x)

    ica = FastICA(n_components=n, random_state=cfg.programs.random_seed, max_iter=500)
    w_ica = ica.fit_transform(x)
    h_ica = ica.components_
    rec_ica = np.linalg.norm(x - w_ica @ h_ica) / np.linalg.norm(x)

    def _sparsity(arr: np.ndarray) -> float:
        return float(np.mean(np.abs(arr) < 1e-3))

    return pd.DataFrame(
        [
            {
                "method": "NMF",
                "reconstruction_error": float(rec_nmf),
                "interpretability_score": float(np.mean(h_nmf >= 0.0)),
                "sparsity_score": _sparsity(h_nmf),
            },
            {"method": "PCA", "reconstruction_error": float(rec_pca), "interpretability_score": 0.0, "sparsity_score": _sparsity(h_pca)},
            {"method": "ICA", "reconstruction_error": float(rec_ica), "interpretability_score": 0.0, "sparsity_score": _sparsity(h_ica)},
        ]
    )


def run_nmf_programs(
    perturb: np.ndarray,
    var_names: list[str],
    cfg: PipelineConfig,
) -> ProgramResults:
    n = max(2, min(cfg.programs.chosen_components, perturb.shape[1] - 1))
    x_nonneg = _nonnegative(perturb - perturb.min())
    nmf = NMF(n_components=n, init="nndsvda", random_state=cfg.programs.random_seed, max_iter=500)
    w = nmf.fit_transform(x_nonneg)
    h = nmf.components_

    scores = pd.DataFrame(w, columns=[f"program_{i}" for i in range(n)])
    loadings = pd.DataFrame(h.T, index=var_names, columns=[f"program_{i}" for i in range(n)])
    selection = compare_factorizations(perturb, cfg)
    return ProgramResults(scores=scores, loadings=loadings, model_selection=selection, chosen_method="NMF")
