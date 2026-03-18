from __future__ import annotations

from pathlib import Path
import tarfile
import tempfile

import numpy as np
import pandas as pd

from .types import DatasetBundle


def _clean_candidates(paths: list[Path]) -> list[Path]:
    return [p for p in paths if "/._" not in str(p) and not p.name.startswith("._")]


def resolve_dataset(path_str: str) -> tuple[Path, str]:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file() and path.suffix == ".h5ad":
        return path, "h5ad"
    if path.is_file() and path.suffix in {".csv", ".tsv"}:
        return path, "tabular"
    if path.is_file() and (path.suffix == ".tar" or path.name.endswith(".tar.gz")):
        return path, "visium_tar"
    if path.is_dir():
        h5ads = _clean_candidates(sorted(path.glob("*.h5ad")))
        if h5ads:
            return h5ads[0], "h5ad"
        visium_h5 = _clean_candidates(sorted(path.glob("**/*filtered_feature_bc_matrix.h5")))
        if visium_h5:
            return visium_h5[0], "visium_h5"
    raise ValueError("Could not auto-detect supported dataset format from input path.")


def _guess_var_names(n_genes: int) -> list[str]:
    return [f"gene_{i}" for i in range(n_genes)]


def _load_10x_h5_fallback(h5_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    try:
        import h5py  # type: ignore
        from scipy.sparse import csc_matrix  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Loading 10x H5 requires either scanpy or (h5py + scipy).") from exc

    with h5py.File(h5_path, "r") as h5:
        grp = h5["matrix"]
        data = grp["data"][:]
        indices = grp["indices"][:]
        indptr = grp["indptr"][:]
        shape = tuple(grp["shape"][:])
        mat = csc_matrix((data, indices, indptr), shape=shape)  # genes x spots
        expr = mat.T.toarray()  # spots x genes

        feat_grp = grp["features"]
        if "name" in feat_grp:
            var_names = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in feat_grp["name"][:]]
        elif "id" in feat_grp:
            var_names = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in feat_grp["id"][:]]
        else:
            var_names = _guess_var_names(expr.shape[1])
        barcodes = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in grp["barcodes"][:]]
    return expr, var_names, barcodes


def load_dataset(resolved: Path, kind: str) -> DatasetBundle:
    if kind == "h5ad":
        try:
            import anndata as ad  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("anndata is required to read .h5ad files. Install anndata.") from exc
        adata = ad.read_h5ad(resolved)
        expr = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        obs = adata.obs.copy()
        if "x" not in obs.columns or "y" not in obs.columns:
            if "spatial" in adata.obsm:
                obs["x"] = adata.obsm["spatial"][:, 0]
                obs["y"] = adata.obsm["spatial"][:, 1]
        var_names = [str(x) for x in adata.var_names]
        return DatasetBundle(expr=expr, obs=obs, var_names=var_names, source_path=resolved, dataset_kind=kind)

    if kind == "tabular":
        sep = "\t" if resolved.suffix == ".tsv" else ","
        df = pd.read_csv(resolved, sep=sep)
        meta_cols = [c for c in ["section_id", "x", "y", "cna_score"] if c in df.columns]
        expr_df = df.drop(columns=meta_cols, errors="ignore")
        expr = expr_df.to_numpy(dtype=float)
        obs = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
        if "section_id" not in obs.columns:
            obs["section_id"] = "section_0"
        return DatasetBundle(
            expr=expr,
            obs=obs,
            var_names=list(expr_df.columns) if len(expr_df.columns) else _guess_var_names(expr.shape[1]),
            source_path=resolved,
            dataset_kind=kind,
        )

    if kind == "visium_h5":
        # Try scanpy first; fallback to direct 10x H5 parsing.
        expr = None
        obs = None
        var_names: list[str] = []
        try:
            import scanpy as sc  # type: ignore

            adata = sc.read_10x_h5(str(resolved))
            expr = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
            obs = adata.obs.copy()
            var_names = [str(x) for x in adata.var_names]
        except Exception:
            expr_fallback, var_names, barcodes = _load_10x_h5_fallback(resolved)
            expr = expr_fallback
            obs = pd.DataFrame({"barcode": barcodes})

        assert expr is not None and obs is not None
        if "section_id" not in obs.columns:
            obs["section_id"] = "section_0"
        if "x" not in obs.columns:
            obs["x"] = np.arange(len(obs), dtype=float)
            obs["y"] = 0.0
        return DatasetBundle(expr=expr, obs=obs, var_names=var_names, source_path=resolved, dataset_kind=kind)

    if kind == "visium_tar":
        if not tarfile.is_tarfile(str(resolved)):
            raise ValueError(f"Input is not a valid tar archive: {resolved}")
        tmp_dir = Path(tempfile.mkdtemp(prefix="omega_visium_"))
        with tarfile.open(resolved, "r:*") as tf:
            tf.extractall(tmp_dir)

        mtx_files = _clean_candidates(sorted(tmp_dir.glob("**/*matrix.mtx*")))
        feat_files = _clean_candidates(sorted(tmp_dir.glob("**/*features.tsv*")))
        bc_files = _clean_candidates(sorted(tmp_dir.glob("**/*barcodes.tsv*")))
        pos_files = _clean_candidates(sorted(tmp_dir.glob("**/*tissue_positions_list.csv")))
        h5_files = _clean_candidates(sorted(tmp_dir.glob("**/*filtered_feature_bc_matrix.h5")))
        if not (mtx_files and feat_files and bc_files):
            if h5_files:
                # Reuse existing H5 route when matrix triple is absent.
                return load_dataset(h5_files[0], "visium_h5")
            raise ValueError("Visium tar missing required matrix/features/barcodes files (and no filtered_feature_bc_matrix.h5 fallback).")

        from scipy.io import mmread  # type: ignore

        matrix = mmread(str(mtx_files[0]))
        matrix = matrix.tocsr() if hasattr(matrix, "tocsr") else matrix
        expr = matrix.T.toarray() if hasattr(matrix.T, "toarray") else np.asarray(matrix.T)

        features = pd.read_csv(feat_files[0], sep="\t", header=None)
        var_names = features.iloc[:, 0].astype(str).tolist()
        barcodes = pd.read_csv(bc_files[0], sep="\t", header=None, names=["barcode"])
        obs = barcodes.copy()
        obs["section_id"] = resolved.stem.replace(".tar", "")

        if pos_files:
            pos = pd.read_csv(pos_files[0], header=None)
            if pos.shape[1] >= 6:
                pos.columns = [
                    "barcode",
                    "in_tissue",
                    "array_row",
                    "array_col",
                    "pxl_row_in_fullres",
                    "pxl_col_in_fullres",
                ]
                obs = obs.merge(pos[["barcode", "pxl_col_in_fullres", "pxl_row_in_fullres"]], on="barcode", how="left")
                obs["x"] = obs["pxl_col_in_fullres"]
                obs["y"] = obs["pxl_row_in_fullres"]
            else:
                obs["x"] = np.arange(len(obs), dtype=float)
                obs["y"] = 0.0
        else:
            obs["x"] = np.arange(len(obs), dtype=float)
            obs["y"] = 0.0

        return DatasetBundle(expr=expr, obs=obs, var_names=var_names, source_path=resolved, dataset_kind=kind)

    raise ValueError(f"Unsupported dataset kind: {kind}")
