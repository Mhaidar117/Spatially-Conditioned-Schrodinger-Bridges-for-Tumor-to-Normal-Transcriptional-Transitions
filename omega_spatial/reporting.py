from __future__ import annotations

from pathlib import Path
import os
import tempfile

# Ensure non-interactive plotting with writable cache in restricted envs.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "omega_mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template


def make_figures(
    out_dir: Path,
    obs: pd.DataFrame,
    perturb_mag: np.ndarray,
    program_scores: pd.DataFrame,
) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    x_col = "x" if "x" in obs.columns else obs.columns[0]
    y_col = "y" if "y" in obs.columns else obs.columns[1]
    plt.figure(figsize=(6, 5))
    plt.scatter(obs[x_col], obs[y_col], c=perturb_mag, s=8, cmap="viridis")
    plt.title("Spatial perturbation magnitude")
    plt.colorbar(label="||u||")
    path1 = fig_dir / "spatial_perturbation_map.png"
    plt.tight_layout()
    plt.savefig(path1, dpi=150)
    plt.close()
    paths.append(str(path1))

    top = program_scores.iloc[:, : min(4, program_scores.shape[1])]
    plt.figure(figsize=(8, 3))
    for col in top.columns:
        plt.plot(top[col].to_numpy(), label=col, alpha=0.8)
    plt.title("Top program scores across spots")
    plt.legend(loc="upper right", fontsize=8)
    path2 = fig_dir / "top_program_scores.png"
    plt.tight_layout()
    plt.savefig(path2, dpi=150)
    plt.close()
    paths.append(str(path2))
    return paths


def _simple_pathway_enrichment(loadings: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    rows = []
    for col in loadings.columns:
        top_genes = loadings[col].sort_values(ascending=False).head(top_n).index.tolist()
        rows.append({"program": col, "pathway": "TopWeightedGenes", "genes": ";".join(top_genes)})
    return pd.DataFrame(rows)


def write_reports(
    out_dir: Path,
    run_name: str,
    readiness: pd.DataFrame,
    qc_summary: pd.DataFrame,
    benchmark: pd.DataFrame,
    programs: pd.DataFrame,
    loadings: pd.DataFrame,
    figures: list[str],
    malignancy_summary: pd.DataFrame,
    malignancy_counts: pd.DataFrame,
) -> None:
    pathway = _simple_pathway_enrichment(loadings)
    pathway.to_csv(out_dir / "pathway_enrichment.csv", index=False)

    template = Template(
        """
<html>
<head><title>{{ run_name }}</title></head>
<body>
<h1>{{ run_name }} - Biologist Interpretation Report</h1>
{% if cna_warning %}
<div style="border:1px solid #cc9900;padding:10px;margin:10px 0;background:#fff9e6;">
<strong>Data warning:</strong> CNA/malignancy scores were not detected in this run.
Tumor/normal marginals cannot be strictly defined by CNA without supplying a CNA column.
</div>
{% endif %}
<h2>Dataset readiness</h2>
{{ readiness_html }}
<h2>QC summary</h2>
{{ qc_html }}
<h2>Benchmark summary</h2>
{{ bench_html }}
<h2>Malignancy scoring summary</h2>
{{ malignancy_summary_html }}
<h2>Malignancy counts by section</h2>
{{ malignancy_counts_html }}
<h2>Top gene programs</h2>
{{ programs_html }}
<h2>Candidate intervention modules</h2>
<p>Programs with highest aggregate perturbation loadings are candidate intervention modules.</p>
<h2>Figures</h2>
{% for fig in figures %}
<div><img src="{{ fig }}" width="700"></div>
{% endfor %}
</body>
</html>
"""
    )
    html = template.render(
        run_name=run_name,
        cna_warning=(
            bool(readiness.get("recommendations", pd.Series(dtype=str)).astype(str).str.contains("Missing CNA", regex=False).any())
            or bool(readiness.get("cna_column", pd.Series(dtype=str)).fillna("N/A").astype(str).str.contains("N/A", regex=False).any())
        ),
        readiness_html=readiness.fillna("N/A").to_html(index=False, na_rep="N/A"),
        qc_html=qc_summary.fillna("N/A").to_html(index=False, na_rep="N/A"),
        bench_html=benchmark.fillna("N/A").to_html(index=False, na_rep="N/A"),
        malignancy_summary_html=malignancy_summary.fillna("N/A").to_html(index=False, na_rep="N/A"),
        malignancy_counts_html=malignancy_counts.fillna("N/A").to_html(index=False, na_rep="N/A"),
        programs_html=programs.head(20).rename_axis("gene").reset_index().fillna("N/A").to_html(index=False, na_rep="N/A"),
        figures=[f"figures/{Path(f).name}" if Path(f).exists() else f for f in figures],
    )
    (out_dir / "report.html").write_text(html, encoding="utf-8")

    # PDF fallback with minimal content if no html->pdf backend.
    pdf_path = out_dir / "report.pdf"
    try:
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.05, 0.95, f"{run_name}\nBiologist Interpretation Summary", va="top", fontsize=14)
            fig.text(0.05, 0.86, "See report.html for full interactive details.", fontsize=10)
            fig.text(0.05, 0.80, benchmark.to_string(index=False)[:2800], family="monospace", fontsize=7)
            pdf.savefig(fig)
            plt.close(fig)
    except Exception:
        pdf_path.write_text("PDF generation fallback. Please open report.html.", encoding="utf-8")
