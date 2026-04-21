from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .types import DatasetBundle, IngestionDiagnostics

if TYPE_CHECKING:
    from .config import PipelineConfig


def _clean_candidates(paths: list[Path]) -> list[Path]:
    return [p for p in paths if "/._" not in str(p) and not p.name.startswith("._")]


def _norm_barcode(val: object) -> str:
    return str(val).strip()


def _visium_outs_has_matrix(outs_dir: Path) -> bool:
    if not outs_dir.is_dir():
        return False
    h5 = outs_dir / "filtered_feature_bc_matrix.h5"
    mtx_dir = outs_dir / "filtered_feature_bc_matrix"
    return h5.is_file() or mtx_dir.is_dir()


def _is_gbm_cohort_root(path: Path) -> bool:
    """True for `GBM_data`-style roots containing per-sample `*/outs` Visium trees."""
    if not path.is_dir() or path.name.startswith("."):
        return False
    if path.name == "GBM_data":
        return True
    subs = [c for c in path.iterdir() if c.is_dir() and not c.name.startswith(".")]
    if len(subs) < 2:
        return False
    vis = sum(1 for c in subs if _visium_outs_has_matrix(c / "outs"))
    return vis >= 2


def _stage1_max_samples() -> int | None:
    raw = os.environ.get("OMEGA_STAGE1_MAX_SAMPLES", "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_cna_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (_repo_root() / p).resolve()


def _resolve_true_cna_inputs(cfg: "PipelineConfig | None") -> tuple[Path | None, Path | None]:
    if cfg is not None:
        rds_path = _resolve_cna_path(cfg.cna.true_score_rds_path) if cfg.cna.true_score_rds_path else None
        regions_path = _resolve_cna_path(cfg.cna.true_score_regions_path) if cfg.cna.true_score_regions_path else None
    else:
        root = _repo_root()
        rds_path = root / "Data" / "Inputs" / "CNA" / "mal_lev.rds"
        regions_path = root / "Data" / "Inputs" / "CNA" / "samples_regions.txt"
    if rds_path is not None and not rds_path.is_file():
        rds_path = None
    if regions_path is not None and not regions_path.is_file():
        regions_path = None
    return rds_path, regions_path


def _load_true_cna_table_from_rds(rds_path: Path) -> pd.DataFrame:
    """
    Read upstream RDS (`mal_lev.rds`) as long-format sample/spot/cna_score.
    Uses Rscript because upstream asset is stored in R native format.
    """
    cmd = [
        "Rscript",
        "-e",
        (
            "args <- commandArgs(trailingOnly=TRUE); "
            "x <- readRDS(args[1]); "
            "rows <- vector('list', length(x)); "
            "nms <- names(x); "
            "for (i in seq_along(x)) { "
            "  v <- as.numeric(x[[i]]); "
            "  b <- names(x[[i]]); "
            "  if (is.null(b)) { b <- rep('', length(v)); }; "
            "  rows[[i]] <- data.frame(sample_key=rep(nms[i], length(v)), spot_id=b, cna_score=v, stringsAsFactors=FALSE); "
            "}; "
            "out <- do.call(rbind, rows); "
            "write.table(out, file='', sep='\\t', quote=FALSE, row.names=FALSE)"
        ),
        str(rds_path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not load true CNA scores because `Rscript` is unavailable."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"Failed to parse true CNA RDS via Rscript: {stderr}") from exc
    df = pd.read_csv(StringIO(proc.stdout), sep="\t")
    if df.empty:
        return pd.DataFrame(columns=["sample_id", "spot_id", "cna_score"])
    df["sample_key"] = df["sample_key"].astype(str).str.strip()
    df["spot_id"] = df["spot_id"].astype(str).map(_norm_barcode)
    df["cna_score"] = pd.to_numeric(df["cna_score"], errors="coerce")
    df = df.dropna(subset=["cna_score"])
    df = df[df["spot_id"] != ""].copy()
    df = df.rename(columns={"sample_key": "sample_id"})
    return df[["sample_id", "spot_id", "cna_score"]]


def _expand_true_cna_with_region_map(cna_df: pd.DataFrame, regions_path: Path | None) -> pd.DataFrame:
    if cna_df.empty or regions_path is None:
        return cna_df
    reg = pd.read_csv(regions_path, sep="\t")
    if not {"cna_samples_name", "sample"}.issubset(reg.columns):
        return cna_df
    reg = reg[["cna_samples_name", "sample"]].dropna().copy()
    reg["cna_samples_name"] = reg["cna_samples_name"].astype(str).str.strip()
    reg["sample"] = reg["sample"].astype(str).str.strip()
    mapped = cna_df.merge(reg, left_on="sample_id", right_on="cna_samples_name", how="inner")
    if mapped.empty:
        return cna_df
    mapped = mapped[["sample", "spot_id", "cna_score"]].rename(columns={"sample": "sample_id"})
    out = pd.concat([cna_df, mapped], ignore_index=True)
    out = out.drop_duplicates(subset=["sample_id", "spot_id"], keep="first")
    return out


def _join_true_cna_scores(
    obs: pd.DataFrame,
    cna_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = obs.copy()
    info: dict[str, object] = {
        "true_cna_joined": False,
        "true_cna_join_match_rate": 0.0,
        "true_cna_join_matches": 0,
        "true_cna_join_key": None,
    }
    if cna_df is None or cna_df.empty or "barcode" not in out.columns:
        return out, info

    keys = [c for c in ("section_id", "metadata_sample", "sample") if c in out.columns]
    if not keys:
        return out, info

    best_col = None
    best_series = None
    best_hits = -1
    cna = cna_df.copy()
    cna["sample_id"] = cna["sample_id"].astype(str).str.strip()
    cna["spot_id"] = cna["spot_id"].astype(str).map(_norm_barcode)
    cna = cna.drop_duplicates(subset=["sample_id", "spot_id"], keep="first")

    for col in keys:
        left = out[[col, "barcode"]].copy()
        left["sample_id"] = left[col].astype(str).str.strip()
        left["spot_id"] = left["barcode"].astype(str).map(_norm_barcode)
        merged = left.merge(cna, on=["sample_id", "spot_id"], how="left")
        score = pd.to_numeric(merged["cna_score"], errors="coerce")
        hits = int(score.notna().sum())
        if hits > best_hits:
            best_hits = hits
            best_col = col
            best_series = score

    if best_series is None or best_hits <= 0:
        return out, info

    existing = pd.to_numeric(out["cna_score"], errors="coerce") if "cna_score" in out.columns else None
    merged_score = best_series if existing is None else best_series.where(best_series.notna(), existing)
    out["cna_score"] = merged_score.to_numpy(dtype=float)
    info["true_cna_joined"] = True
    info["true_cna_join_key"] = f"{best_col}+barcode"
    info["true_cna_join_matches"] = best_hits
    info["true_cna_join_match_rate"] = best_hits / max(len(out), 1)
    return out, info


def _build_true_cna_table(cfg: "PipelineConfig | None") -> pd.DataFrame | None:
    rds_path, regions_path = _resolve_true_cna_inputs(cfg)
    if rds_path is None:
        return None
    cna_df = _load_true_cna_table_from_rds(rds_path)
    return _expand_true_cna_with_region_map(cna_df, regions_path)


def discover_visium_metadata_path(anchor: Path) -> Path | None:
    """
    Locate `visium_metadata.csv` next to the GBM cohort (`general/visium_metadata.csv`).
    Override with env `OMEGA_VISIUM_METADATA` (absolute or relative path).
    """
    env = os.environ.get("OMEGA_VISIUM_METADATA", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_file() else None
    p = anchor.expanduser().resolve()
    if p.name == "outs":
        p = p.parent
    if p.name == "GBM_data":
        cand = p.parent / "visium_metadata.csv"
        return cand if cand.is_file() else None
    if (p / "outs").is_dir() and p.parent.name == "GBM_data":
        cand = p.parent.parent / "visium_metadata.csv"
        return cand if cand.is_file() else None
    cand = p.parent / "visium_metadata.csv"
    return cand if cand.is_file() else None


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
        # Greenwald-style layout: detect before generic **/*.h5 glob (would pick one sample only).
        if _is_gbm_cohort_root(path):
            return path, "gbm_cohort"
        if (path / "outs").is_dir() and _visium_outs_has_matrix(path / "outs"):
            return path, "gbm_visium_sample"
        if _visium_outs_has_matrix(path):
            return path, "visium_outs"
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


def _load_visium_mtx_folder(mtx_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    from scipy.io import mmread  # type: ignore

    mtx_files = sorted(mtx_dir.glob("matrix.mtx*"))
    feat_files = sorted(mtx_dir.glob("features.tsv*"))
    bc_files = sorted(mtx_dir.glob("barcodes.tsv*"))
    if not (mtx_files and feat_files and bc_files):
        raise FileNotFoundError(f"Incomplete filtered_feature_bc_matrix folder: {mtx_dir}")
    matrix = mmread(str(mtx_files[0]))
    matrix = matrix.tocsr() if hasattr(matrix, "tocsr") else matrix
    expr = matrix.T.toarray() if hasattr(matrix.T, "toarray") else np.asarray(matrix.T)
    features = pd.read_csv(feat_files[0], sep="\t", header=None)
    var_names = features.iloc[:, 0].astype(str).tolist()
    bc = pd.read_csv(bc_files[0], sep="\t", header=None, names=["barcode"])
    barcodes = bc["barcode"].astype(str).tolist()
    return expr, var_names, barcodes


def _read_tissue_positions(outs_dir: Path) -> pd.DataFrame | None:
    spatial = outs_dir / "spatial"
    if not spatial.is_dir():
        return None
    pos_files = _clean_candidates(sorted(spatial.glob("tissue_positions*.csv")))
    if not pos_files:
        return None
    pos = pd.read_csv(pos_files[0], header=None)
    if pos.shape[1] < 6:
        return None
    pos.columns = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    pos["barcode"] = pos["barcode"].map(_norm_barcode)
    return pos


def _load_expression_from_outs(outs_dir: Path) -> tuple[np.ndarray, list[str], pd.DataFrame, list[str]]:
    """Returns expr (spots x genes), var_names, obs with `barcode`, and provenance file paths."""
    sources: list[str] = []
    h5_path = outs_dir / "filtered_feature_bc_matrix.h5"
    mtx_dir = outs_dir / "filtered_feature_bc_matrix"
    expr: np.ndarray | None = None
    var_names: list[str] = []
    obs: pd.DataFrame | None = None

    if h5_path.is_file():
        sources.append(str(h5_path))
        try:
            import scanpy as sc  # type: ignore

            adata = sc.read_10x_h5(str(h5_path))
            expr = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
            var_names = [str(x) for x in adata.var_names]
            obs = adata.obs.copy()
            if "barcode" in obs.columns:
                obs["barcode"] = obs["barcode"].astype(str).map(_norm_barcode)
            else:
                obs.insert(0, "barcode", obs.index.astype(str).map(_norm_barcode))
        except Exception:
            expr_fb, var_names, barcodes = _load_10x_h5_fallback(h5_path)
            expr = expr_fb
            obs = pd.DataFrame({"barcode": [_norm_barcode(b) for b in barcodes]})
    elif mtx_dir.is_dir():
        sources.append(str(mtx_dir))
        expr_fb, var_names, barcodes = _load_visium_mtx_folder(mtx_dir)
        expr = expr_fb
        obs = pd.DataFrame({"barcode": [_norm_barcode(b) for b in barcodes]})
    else:
        raise FileNotFoundError(f"No Visium matrix under outs: {outs_dir}")

    assert expr is not None and obs is not None
    pos = _read_tissue_positions(outs_dir)
    if pos is not None:
        sources.append(str(outs_dir / "spatial"))
        obs = obs.merge(
            pos[["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]],
            on="barcode",
            how="left",
        )
        obs["pxl_col_in_fullres"] = pd.to_numeric(obs["pxl_col_in_fullres"], errors="coerce")
        obs["pxl_row_in_fullres"] = pd.to_numeric(obs["pxl_row_in_fullres"], errors="coerce")
    else:
        obs["pxl_col_in_fullres"] = np.nan
        obs["pxl_row_in_fullres"] = np.nan

    return expr, var_names, obs, sources


EVAL_METADATA_COLUMNS = ("mp", "layer", "ivygap", "org1", "org2", "cc", "cna_bin")
OPTIONAL_MALIGNANCY_COLUMNS = ("cna_score", "malignancy_score")


def join_visium_metadata(
    obs: pd.DataFrame,
    meta: pd.DataFrame | None,
    section_id: str,
    *,
    sample_col: str = "sample",
    spot_col: str = "spot_id",
) -> tuple[pd.DataFrame, dict[str, float | str | bool]]:
    """
    Left-join cohort metadata onto Visium obs using normalized `spot_id` <-> `barcode`.
    Canonical `x`,`y` use `centroid_x`,`centroid_y` when the join hits; otherwise Space Ranger pixels.
    """
    out = obs.copy()
    out["section_id"] = str(section_id)
    stats: dict[str, float | str | bool] = {"section_id": section_id, "join_key_used": f"{spot_col}<->barcode"}

    if meta is None or meta.empty:
        out["metadata_joined"] = False
        out["x"] = pd.to_numeric(out["pxl_col_in_fullres"], errors="coerce")
        out["y"] = pd.to_numeric(out["pxl_row_in_fullres"], errors="coerce")
        mask = out["x"].isna()
        out.loc[mask, "x"] = np.arange(mask.sum(), dtype=float)
        out.loc[mask, "y"] = 0.0
        stats["match_rate"] = 0.0
        stats["n_matched"] = 0
        return out, stats

    if sample_col not in meta.columns or spot_col not in meta.columns:
        raise ValueError(f"visium_metadata must contain '{sample_col}' and '{spot_col}'.")

    sub = meta[meta[sample_col].astype(str) == str(section_id)].copy()
    sub["_join_key"] = sub[spot_col].map(_norm_barcode)
    out["_join_key"] = out["barcode"].map(_norm_barcode)

    right = sub.rename(columns={sample_col: "metadata_sample"})
    merged = out.merge(right, on="_join_key", how="left", suffixes=("", "_meta_dup"))
    merged.drop(columns=["_join_key"], inplace=True)

    if "centroid_x" in merged.columns and "centroid_y" in merged.columns:
        cx = pd.to_numeric(merged["centroid_x"], errors="coerce")
        cy = pd.to_numeric(merged["centroid_y"], errors="coerce")
        hit = cx.notna() & cy.notna()
        merged["metadata_joined"] = hit.to_numpy()
        merged["x"] = np.where(hit, cx, pd.to_numeric(merged["pxl_col_in_fullres"], errors="coerce"))
        merged["y"] = np.where(hit, cy, pd.to_numeric(merged["pxl_row_in_fullres"], errors="coerce"))
    else:
        merged["metadata_joined"] = False
        merged["x"] = pd.to_numeric(merged["pxl_col_in_fullres"], errors="coerce")
        merged["y"] = pd.to_numeric(merged["pxl_row_in_fullres"], errors="coerce")

    # Fallback row index for missing spatial
    bad = merged["x"].isna()
    if bad.any():
        merged.loc[bad, "x"] = np.arange(int(bad.sum()), dtype=float)
        merged.loc[bad, "y"] = 0.0

    n_hit = int(merged["metadata_joined"].sum()) if "metadata_joined" in merged.columns else 0
    rate = n_hit / max(len(merged), 1)
    stats["match_rate"] = rate
    stats["n_matched"] = float(n_hit)
    return merged, stats


def _expected_metadata_columns(meta: pd.DataFrame | None) -> list[str]:
    if meta is None or meta.empty:
        return []
    present = set(meta.columns.astype(str))
    missing = [c for c in EVAL_METADATA_COLUMNS if c not in present]
    missing.extend([c for c in OPTIONAL_MALIGNANCY_COLUMNS if c not in present])
    return missing


def load_visium_sample_bundle(
    sample_dir: Path,
    *,
    section_id: str | None = None,
    metadata_path: Path | None = None,
    meta_df: pd.DataFrame | None = None,
    true_cna_df: pd.DataFrame | None = None,
) -> tuple[DatasetBundle, IngestionDiagnostics]:
    """
    Load one sample directory containing `outs/` (Greenwald-style layout).
    """
    outs = sample_dir / "outs"
    if not _visium_outs_has_matrix(outs):
        raise FileNotFoundError(f"Expected Visium outs with matrix under: {outs}")

    sid = section_id or sample_dir.name
    expr, var_names, obs, src_files = _load_expression_from_outs(outs)

    meta_path = metadata_path
    if meta_df is None and meta_path is None:
        meta_path = discover_visium_metadata_path(sample_dir)
    if meta_df is None and meta_path is not None:
        meta_df = pd.read_csv(meta_path)

    obs_joined, st = join_visium_metadata(obs, meta_df, sid)
    obs_joined, cna_join_stats = _join_true_cna_scores(obs_joined, true_cna_df)
    warnings: list[str] = []
    if float(st.get("match_rate", 0)) < 1.0:
        warnings.append(
            f"Section {sid}: metadata match rate {float(st.get('match_rate', 0)):.4f} "
            f"({int(st.get('n_matched', 0))}/{len(obs_joined)} spots)."
        )
    cna_join_rate = float(cna_join_stats.get("true_cna_join_match_rate", 0.0))
    if bool(cna_join_stats.get("true_cna_joined")) and cna_join_rate < 1.0:
        warnings.append(
            f"Section {sid}: true CNA join coverage {cna_join_rate:.4f} "
            f"({int(cna_join_stats.get('true_cna_join_matches', 0))}/{len(obs_joined)} spots)."
        )

    missing_cols = _expected_metadata_columns(meta_df)
    diag = IngestionDiagnostics(
        join_keys_attempted=[f"{st.get('join_key_used', 'spot_id<->barcode')}"],
        join_key_used=str(st.get("join_key_used", "spot_id<->barcode")),
        metadata_path=str(meta_path) if meta_path else None,
        metadata_rows_matched=int(st.get("n_matched", 0)),
        metadata_match_rate=float(st.get("match_rate", 0.0)),
        samples_loaded=[sid],
        source_files=src_files + ([str(meta_path)] if meta_path else []),
        missing_metadata_columns=missing_cols,
        warnings=warnings,
        per_sample_match_rates={sid: float(st.get("match_rate", 0.0))},
    )
    obs_joined.attrs["omega_ingestion"] = asdict(diag)
    return (
        DatasetBundle(
            expr=expr,
            obs=obs_joined,
            var_names=var_names,
            source_path=sample_dir.resolve(),
            dataset_kind="gbm_visium_sample",
        ),
        diag,
    )


def load_gbm_cohort_bundle(
    gbm_data_dir: Path,
    *,
    metadata_path: Path | None = None,
    meta_df: pd.DataFrame | None = None,
    max_samples: int | None = None,
    cfg: "PipelineConfig | None" = None,
) -> tuple[DatasetBundle, IngestionDiagnostics]:
    """
    Load every sample under `GBM_data/<sample>/outs`, concatenate along spots.
    """
    limit = max_samples if max_samples is not None else _stage1_max_samples()
    meta_path = metadata_path
    if meta_df is None:
        if meta_path is None:
            meta_path = discover_visium_metadata_path(gbm_data_dir)
        if meta_path is not None:
            meta_df = pd.read_csv(meta_path)
    true_cna_df = _build_true_cna_table(cfg)
    cna_rds_path, cna_regions_path = _resolve_true_cna_inputs(cfg)

    samples = sorted(
        [c for c in gbm_data_dir.iterdir() if c.is_dir() and not c.name.startswith(".") and _visium_outs_has_matrix(c / "outs")],
        key=lambda p: p.name.lower(),
    )
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError(f"No Visium samples found under: {gbm_data_dir}")

    bundles: list[DatasetBundle] = []
    diags: list[IngestionDiagnostics] = []
    for sdir in samples:
        b, d = load_visium_sample_bundle(
            sdir,
            metadata_path=meta_path,
            meta_df=meta_df,
            true_cna_df=true_cna_df,
        )
        bundles.append(b)
        diags.append(d)

    var0 = bundles[0].var_names
    for b in bundles[1:]:
        if b.var_names != var0:
            raise ValueError(
                "Gene lists differ across GBM samples; cannot concatenate. "
                f"First sample n_genes={len(var0)}, other sample n_genes={len(b.var_names)}."
            )

    expr = np.vstack([b.expr for b in bundles])
    obs = pd.concat([b.obs for b in bundles], ignore_index=True)

    n_total = len(obs)
    n_matched = int(obs["metadata_joined"].sum()) if "metadata_joined" in obs.columns else 0
    uniq_sources: list[str] = []
    for d in diags:
        for p in d.source_files:
            if p not in uniq_sources:
                uniq_sources.append(p)

    combined = IngestionDiagnostics(
        join_keys_attempted=["spot_id<->barcode"],
        join_key_used="spot_id<->barcode",
        metadata_path=str(meta_path) if meta_path else None,
        metadata_rows_matched=n_matched,
        metadata_match_rate=n_matched / max(n_total, 1),
        samples_loaded=[d.samples_loaded[0] for d in diags],
        source_files=uniq_sources
        + ([str(cna_rds_path)] if cna_rds_path is not None else [])
        + ([str(cna_regions_path)] if cna_regions_path is not None else []),
        missing_metadata_columns=diags[0].missing_metadata_columns if diags else [],
        warnings=[w for d in diags for w in d.warnings],
        per_sample_match_rates={k: v for d in diags for k, v in d.per_sample_match_rates.items()},
    )
    obs.attrs["omega_ingestion"] = asdict(combined)
    return (
        DatasetBundle(
            expr=expr,
            obs=obs,
            var_names=var0,
            source_path=gbm_data_dir.resolve(),
            dataset_kind="gbm_cohort",
        ),
        combined,
    )


def load_dataset(resolved: Path, kind: str, cfg: "PipelineConfig | None" = None) -> DatasetBundle:
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

    if kind == "gbm_cohort":
        bundle, _diag = load_gbm_cohort_bundle(resolved, cfg=cfg)
        return bundle

    if kind == "gbm_visium_sample":
        true_cna_df = _build_true_cna_table(cfg)
        bundle, _diag = load_visium_sample_bundle(resolved, true_cna_df=true_cna_df)
        return bundle

    if kind == "visium_outs":
        # resolved points at `.../outs` directory
        parent = resolved.parent
        true_cna_df = _build_true_cna_table(cfg)
        bundle, _diag = load_visium_sample_bundle(parent, true_cna_df=true_cna_df)
        return DatasetBundle(
            expr=bundle.expr,
            obs=bundle.obs,
            var_names=bundle.var_names,
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
                return load_dataset(h5_files[0], "visium_h5", cfg=cfg)
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
