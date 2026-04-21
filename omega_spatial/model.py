from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .config import BridgeConfig, PipelineConfig


@dataclass
class BridgeTrainingSummary:
    """Fitting diagnostics for the conditional drift model."""

    n_spots: int
    n_genes: int
    n_normal: int
    n_tumor: int
    n_intermediate: int
    n_other_label: int
    ridge_lambda: float
    spatial_smoothing_alpha: float
    reverse_step_size: float
    default_transport_n_steps: int
    residual_frobenius: float
    mean_residual_l2_per_spot: float
    normal_reference_l2_norm: float
    backend: str = "linear"
    train_loss_final: float = float("nan")
    train_loss_best: float = float("nan")
    train_steps_completed: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class TransportModel(Protocol):
    backend: str
    alpha: float
    normal_reference: np.ndarray
    reverse_step_size: float
    default_n_steps: int
    training_summary: BridgeTrainingSummary

    def score(self, x_t: np.ndarray, context: np.ndarray, t: float | np.ndarray | None = None) -> np.ndarray: ...

    def reverse_step(
        self,
        x_t: np.ndarray,
        context: np.ndarray,
        step_size: float | None = None,
        t: float | np.ndarray | None = None,
    ) -> np.ndarray: ...

    def transport(
        self,
        expr: np.ndarray,
        context: np.ndarray,
        *,
        n_steps: int | None = None,
        step_size: float | None = None,
    ) -> np.ndarray: ...


@dataclass
class SpatialBridgeModel:
    """
    Spatially conditioned drift field: score(x, c) approximates the vector field that
    pulls expression toward the normal reference mean, modulated by neighbor context.
    """

    w_expr: np.ndarray
    w_ctx: np.ndarray
    bias: np.ndarray
    alpha: float
    normal_reference: np.ndarray
    reverse_step_size: float
    default_n_steps: int
    ridge_lambda: float
    training_summary: BridgeTrainingSummary
    backend: str = "linear"

    def score(self, x_t: np.ndarray, context: np.ndarray, t: float | np.ndarray | None = None) -> np.ndarray:
        return x_t @ self.w_expr + context @ self.w_ctx + self.bias

    def reverse_step(
        self,
        x_t: np.ndarray,
        context: np.ndarray,
        step_size: float | None = None,
        t: float | np.ndarray | None = None,
    ) -> np.ndarray:
        ss = self.reverse_step_size if step_size is None else step_size
        return x_t + ss * self.score(x_t, context, t=t) + self.alpha * (context - x_t)

    def transport(
        self,
        expr: np.ndarray,
        context: np.ndarray,
        *,
        n_steps: int | None = None,
        step_size: float | None = None,
    ) -> np.ndarray:
        return generate_counterfactuals(self, expr, context, n_steps=n_steps, step_size=step_size)


@dataclass
class BayesianRidgeBridgeModel:
    """Ridge bridge with a closed-form Gaussian posterior over weights.

    The mean prediction is identical to :class:`SpatialBridgeModel` — we fit
    exactly the same ridge regression on the same features — but we additionally
    store the posterior precision-inverse ``M = (Z^T Z + lam I)^{-1}`` and an
    estimated pooled residual variance ``sigma^2``.  Predictive drift variance
    at a new design row ``z_*`` is ``sigma^2 * z_* M z_*^T`` (a single scalar
    per spot; ridge treats output genes independently under shared noise).
    """

    w_expr: np.ndarray
    w_ctx: np.ndarray
    bias: np.ndarray
    alpha: float
    normal_reference: np.ndarray
    reverse_step_size: float
    default_n_steps: int
    ridge_lambda: float
    training_summary: BridgeTrainingSummary
    precision_inv: np.ndarray
    noise_var: float
    backend: str = "bayesian_linear"

    def score(self, x_t: np.ndarray, context: np.ndarray, t: float | np.ndarray | None = None) -> np.ndarray:
        return x_t @ self.w_expr + context @ self.w_ctx + self.bias

    def reverse_step(
        self,
        x_t: np.ndarray,
        context: np.ndarray,
        step_size: float | None = None,
        t: float | np.ndarray | None = None,
    ) -> np.ndarray:
        ss = self.reverse_step_size if step_size is None else step_size
        return x_t + ss * self.score(x_t, context, t=t) + self.alpha * (context - x_t)

    def transport(
        self,
        expr: np.ndarray,
        context: np.ndarray,
        *,
        n_steps: int | None = None,
        step_size: float | None = None,
    ) -> np.ndarray:
        return generate_counterfactuals(self, expr, context, n_steps=n_steps, step_size=step_size)

    def predictive_drift_std(self, x_t: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Per-spot scalar std of the drift prediction under the ridge posterior."""
        x = np.asarray(x_t, dtype=float)
        c = np.asarray(context, dtype=float)
        if x.shape != c.shape:
            raise ValueError(f"x_t/context shape mismatch: {x.shape} vs {c.shape}")
        n = x.shape[0]
        z = np.concatenate([x, c, np.ones((n, 1))], axis=1)
        mz = z @ self.precision_inv
        var = self.noise_var * np.einsum("ij,ij->i", mz, z)
        return np.sqrt(np.maximum(var, 0.0))

    def predictive_perturbation_std(
        self,
        x_t: np.ndarray,
        context: np.ndarray,
        *,
        n_steps: int | None = None,
        step_size: float | None = None,
    ) -> np.ndarray:
        """Scalar per-spot std on the integrated perturbation norm.

        Conservative linear propagation: treats the drift posterior as constant
        over the short integration window, so the total perturbation magnitude
        inherits a std of ``drift_std * |step_size| * n_steps``.  This is an
        upper bound for well-conditioned problems and is the right order of
        magnitude for uncertainty triage.
        """
        ns = int(self.default_n_steps if n_steps is None else n_steps)
        ss = float(self.reverse_step_size if step_size is None else step_size)
        drift_std = self.predictive_drift_std(x_t, context)
        return drift_std * abs(ss) * max(ns, 1)


@dataclass
class NeuralTransportModel:
    """
    Stage 4b neural transport backend with a learned drift field:
    drift = f_theta([x_t, context, time_embedding(t)]).
    """

    net: Any
    alpha: float
    normal_reference: np.ndarray
    reverse_step_size: float
    default_n_steps: int
    ridge_lambda: float
    training_summary: BridgeTrainingSummary
    device: str = "cpu"
    backend: str = "neural"

    def _time_features(self, n: int, t: float | np.ndarray | None) -> np.ndarray:
        if t is None:
            tt = np.zeros((n, 1), dtype=float)
        elif np.isscalar(t):
            tt = np.full((n, 1), float(t), dtype=float)
        else:
            arr = np.asarray(t, dtype=float).reshape(-1, 1)
            if arr.shape[0] != n:
                raise ValueError(f"time vector length mismatch: {arr.shape[0]} vs {n}")
            tt = arr
        return np.concatenate([tt, np.sin(2 * np.pi * tt), np.cos(2 * np.pi * tt)], axis=1)

    def score(self, x_t: np.ndarray, context: np.ndarray, t: float | np.ndarray | None = None) -> np.ndarray:
        x_arr = np.asarray(x_t, dtype=float)
        c_arr = np.asarray(context, dtype=float)
        if x_arr.shape != c_arr.shape:
            raise ValueError(f"x_t/context shape mismatch: {x_arr.shape} vs {c_arr.shape}")
        n = x_arr.shape[0]
        tf = self._time_features(n, t)
        feat = np.concatenate([x_arr, c_arr, tf], axis=1)
        try:
            import torch
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError("PyTorch is required for neural transport inference.") from ex
        self.net.eval()
        with torch.no_grad():
            xin = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
            out = self.net(xin).detach().cpu().numpy()
        return np.asarray(out, dtype=float)

    def reverse_step(
        self,
        x_t: np.ndarray,
        context: np.ndarray,
        step_size: float | None = None,
        t: float | np.ndarray | None = None,
    ) -> np.ndarray:
        ss = self.reverse_step_size if step_size is None else step_size
        x_arr = np.asarray(x_t, dtype=float)
        c_arr = np.asarray(context, dtype=float)
        drift = self.score(x_arr, c_arr, t=t)
        return x_arr + ss * drift + self.alpha * (c_arr - x_arr)

    def transport(
        self,
        expr: np.ndarray,
        context: np.ndarray,
        *,
        n_steps: int | None = None,
        step_size: float | None = None,
    ) -> np.ndarray:
        return generate_counterfactuals(self, expr, context, n_steps=n_steps, step_size=step_size)


def _label_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lab = np.asarray(labels).astype(str)
    normal_idx = np.flatnonzero(lab == "normal")
    tumor_idx = np.flatnonzero(lab == "tumor")
    inter_idx = np.flatnonzero(lab == "intermediate")
    known = np.zeros(lab.shape[0], dtype=bool)
    for idx in (normal_idx, tumor_idx, inter_idx):
        known[idx] = True
    other_idx = np.flatnonzero(~known)
    return normal_idx, tumor_idx, inter_idx, other_idx


def _fit_ridge_multi_target(
    z: np.ndarray,
    target: np.ndarray,
    ridge_lambda: float,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    """Solve min ||W^{1/2}(Z w - y)||_F^2 + lam ||w||_F^2 for w shape (p, d)."""
    n, p = z.shape
    _, d = target.shape
    if sample_weight is None:
        ztz = z.T @ z
        zty = z.T @ target
    else:
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError("sample_weight must have length n_spots")
        zw = z * w[:, np.newaxis]
        ztz = z.T @ zw
        zty = zw.T @ target
    a = ztz + ridge_lambda * np.eye(p)
    return np.linalg.solve(a, zty)


def train_spatial_bridge(
    expr: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    cfg: PipelineConfig,
    sample_weight: np.ndarray | None = None,
) -> SpatialBridgeModel:
    """
    Fit a linear conditional score by ridge regression toward the normal reference mean.

    Target per spot: (normal_mu - x_i). Context enters as auxiliary predictors so the
    learned field is spatially modulated without reimplementing Stage 3 graph logic.
    """
    x = np.asarray(expr, dtype=float)
    c = np.asarray(context, dtype=float)
    if x.shape != c.shape:
        raise ValueError(f"expr and context must have the same shape; got {x.shape} vs {c.shape}")
    n, dim = x.shape
    bridge_cfg: BridgeConfig = cfg.bridge
    lam = float(bridge_cfg.ridge_lambda)

    normal_idx, tumor_idx, inter_idx, other_idx = _label_indices(labels)
    warnings: list[str] = []
    if normal_idx.size == 0:
        warnings.append("No spots labeled 'normal'; using full cohort for reference mean.")
        ref_idx = np.arange(n)
    else:
        ref_idx = normal_idx

    normal_mu = x[ref_idx].mean(axis=0, keepdims=True)
    target_drift = normal_mu - x

    z = np.concatenate([x, c, np.ones((n, 1))], axis=1)
    w_mat = _fit_ridge_multi_target(z, target_drift, lam, sample_weight)

    pred = z @ w_mat
    resid = target_drift - pred
    residual_fro = float(np.linalg.norm(resid, ord="fro"))
    mean_res_l2 = float(np.linalg.norm(resid, axis=1).mean())

    w_expr = w_mat[:dim, :]
    w_ctx = w_mat[dim : 2 * dim, :]
    bias = w_mat[-1, :]

    summary = BridgeTrainingSummary(
        n_spots=n,
        n_genes=dim,
        n_normal=int(normal_idx.size),
        n_tumor=int(tumor_idx.size),
        n_intermediate=int(inter_idx.size),
        n_other_label=int(other_idx.size),
        ridge_lambda=lam,
        spatial_smoothing_alpha=float(cfg.spatial.spatial_smoothing_alpha),
        reverse_step_size=float(bridge_cfg.reverse_step_size),
        default_transport_n_steps=int(bridge_cfg.transport_n_steps),
        residual_frobenius=residual_fro,
        mean_residual_l2_per_spot=mean_res_l2,
        normal_reference_l2_norm=float(np.linalg.norm(normal_mu)),
        warnings=warnings,
    )

    return SpatialBridgeModel(
        w_expr=w_expr,
        w_ctx=w_ctx,
        bias=bias,
        alpha=float(cfg.spatial.spatial_smoothing_alpha),
        normal_reference=normal_mu,
        reverse_step_size=float(bridge_cfg.reverse_step_size),
        default_n_steps=int(bridge_cfg.transport_n_steps),
        ridge_lambda=lam,
        training_summary=summary,
    )


def _train_bayesian_linear_bridge(
    expr: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    cfg: PipelineConfig,
    sample_weight: np.ndarray | None = None,
) -> BayesianRidgeBridgeModel:
    """Fit the ridge bridge and retain the closed-form Gaussian posterior.

    Delegates the mean-prediction fit to the same code path as the deterministic
    linear backend, then recomputes ``(Z^T Z + lam I)^{-1}`` and a pooled noise
    variance ``sigma^2`` from the residuals.  Storing the precision inverse is
    ``O(p^2)`` memory (p = 2*n_genes + 1) and makes predictive-variance queries
    cheap at inference time.
    """
    base = train_spatial_bridge(expr, context, labels, cfg, sample_weight=sample_weight)

    x = np.asarray(expr, dtype=float)
    c = np.asarray(context, dtype=float)
    n, dim = x.shape
    lam = float(cfg.bridge.ridge_lambda)
    z = np.concatenate([x, c, np.ones((n, 1))], axis=1)
    p = z.shape[1]

    if sample_weight is None:
        ztz = z.T @ z
    else:
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
        ztz = z.T @ (z * w[:, None])

    precision_inv = np.linalg.inv(ztz + lam * np.eye(p))

    # Reconstruct residuals via the same (mean) fit that produced base.
    w_mat = np.concatenate(
        [base.w_expr, base.w_ctx, base.bias.reshape(1, -1)], axis=0
    )
    residuals = (base.normal_reference - x) - z @ w_mat
    # Pool across spots and genes — ridge shares a noise variance across outputs.
    denom = max(1, (n * dim) - p)
    noise_var = float(np.sum(residuals**2) / denom)

    summary = base.training_summary
    summary.backend = "bayesian_linear"

    return BayesianRidgeBridgeModel(
        w_expr=base.w_expr,
        w_ctx=base.w_ctx,
        bias=base.bias,
        alpha=base.alpha,
        normal_reference=base.normal_reference,
        reverse_step_size=base.reverse_step_size,
        default_n_steps=base.default_n_steps,
        ridge_lambda=base.ridge_lambda,
        training_summary=summary,
        precision_inv=precision_inv,
        noise_var=noise_var,
    )


def _train_neural_transport(
    expr: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    cfg: PipelineConfig,
    sample_weight: np.ndarray | None = None,
) -> NeuralTransportModel:
    try:
        import torch
        from torch import nn
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(
            "bridge.backend='neural' requires PyTorch. Install torch to enable Stage 4b neural transport."
        ) from ex

    x = np.asarray(expr, dtype=float)
    c = np.asarray(context, dtype=float)
    if x.shape != c.shape:
        raise ValueError(f"expr and context must have same shape; got {x.shape} vs {c.shape}")
    n, dim = x.shape
    bridge_cfg: BridgeConfig = cfg.bridge
    normal_idx, tumor_idx, inter_idx, other_idx = _label_indices(labels)
    warnings: list[str] = []
    if normal_idx.size == 0:
        warnings.append("No spots labeled 'normal'; using full cohort for reference mean.")
        ref_idx = np.arange(n)
    else:
        ref_idx = normal_idx
    normal_mu = x[ref_idx].mean(axis=0, keepdims=True)
    target_drift = normal_mu - x

    hidden_dim = int(bridge_cfg.neural_hidden_dim)
    depth = max(1, int(bridge_cfg.neural_num_layers))
    dropout = float(bridge_cfg.neural_dropout)
    in_dim = (2 * dim) + 3
    layers: list[nn.Module] = []
    cur = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(cur, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        cur = hidden_dim
    layers.append(nn.Linear(cur, dim))
    net = nn.Sequential(*layers)

    requested_device = str(bridge_cfg.neural_device).strip().lower()
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested_device
    net = net.to(device)
    opt = torch.optim.AdamW(
        net.parameters(),
        lr=float(bridge_cfg.neural_learning_rate),
        weight_decay=float(bridge_cfg.neural_weight_decay),
    )

    batch_size = max(16, int(cfg.train.batch_size))
    max_steps = max(1, int(bridge_cfg.neural_train_steps or cfg.train.steps))
    rng = np.random.default_rng(int(cfg.train.random_seed))
    w_arr = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if w_arr is not None and w_arr.shape[0] != n:
        raise ValueError("sample_weight must have length n_spots")

    def _time_feats(tt: np.ndarray) -> np.ndarray:
        tcol = tt.reshape(-1, 1)
        return np.concatenate([tcol, np.sin(2 * np.pi * tcol), np.cos(2 * np.pi * tcol)], axis=1)

    loss_trace: list[float] = []
    net.train()
    for _ in range(max_steps):
        idx = rng.integers(0, n, size=min(batch_size, n))
        x_b = x[idx]
        c_b = c[idx]
        y_b = target_drift[idx]
        t = rng.random(size=(x_b.shape[0],), dtype=float)
        x_t = x_b + t[:, None] * (normal_mu - x_b)
        feat = np.concatenate([x_t, c_b, _time_feats(t)], axis=1)
        xin = torch.as_tensor(feat, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_b, dtype=torch.float32, device=device)
        pred = net(xin)
        if w_arr is None:
            loss = torch.mean((pred - y) ** 2)
        else:
            wb = torch.as_tensor(w_arr[idx], dtype=torch.float32, device=device).reshape(-1, 1)
            loss = torch.mean(wb * ((pred - y) ** 2))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_trace.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        t0 = np.zeros((n,), dtype=float)
        feat_all = np.concatenate([x, c, _time_feats(t0)], axis=1)
        pred_all = net(torch.as_tensor(feat_all, dtype=torch.float32, device=device)).detach().cpu().numpy()
    resid = target_drift - pred_all
    residual_fro = float(np.linalg.norm(resid, ord="fro"))
    mean_res_l2 = float(np.linalg.norm(resid, axis=1).mean())

    summary = BridgeTrainingSummary(
        n_spots=n,
        n_genes=dim,
        n_normal=int(normal_idx.size),
        n_tumor=int(tumor_idx.size),
        n_intermediate=int(inter_idx.size),
        n_other_label=int(other_idx.size),
        ridge_lambda=float(bridge_cfg.ridge_lambda),
        spatial_smoothing_alpha=float(cfg.spatial.spatial_smoothing_alpha),
        reverse_step_size=float(bridge_cfg.reverse_step_size),
        default_transport_n_steps=int(bridge_cfg.transport_n_steps),
        residual_frobenius=residual_fro,
        mean_residual_l2_per_spot=mean_res_l2,
        normal_reference_l2_norm=float(np.linalg.norm(normal_mu)),
        backend="neural",
        train_loss_final=float(loss_trace[-1]) if loss_trace else float("nan"),
        train_loss_best=float(np.nanmin(loss_trace)) if loss_trace else float("nan"),
        train_steps_completed=int(max_steps),
        warnings=warnings,
    )
    return NeuralTransportModel(
        net=net,
        alpha=float(cfg.spatial.spatial_smoothing_alpha),
        normal_reference=normal_mu,
        reverse_step_size=float(bridge_cfg.reverse_step_size),
        default_n_steps=int(bridge_cfg.transport_n_steps),
        ridge_lambda=float(bridge_cfg.ridge_lambda),
        training_summary=summary,
        device=device,
    )


def train_transport_backend(
    expr: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    cfg: PipelineConfig,
    sample_weight: np.ndarray | None = None,
) -> TransportModel:
    backend = str(getattr(cfg.bridge, "backend", "linear")).strip().lower()
    if backend == "linear":
        return train_spatial_bridge(expr, context, labels, cfg, sample_weight=sample_weight)
    if backend == "bayesian_linear":
        return _train_bayesian_linear_bridge(expr, context, labels, cfg, sample_weight=sample_weight)
    if backend == "neural":
        return _train_neural_transport(expr, context, labels, cfg, sample_weight=sample_weight)
    raise ValueError(f"Unsupported bridge backend: {backend}")


def generate_counterfactuals(
    model: TransportModel,
    expr: np.ndarray,
    context: np.ndarray,
    n_steps: int | None = None,
    step_size: float | None = None,
) -> np.ndarray:
    """
    Integrate the learned reverse-time drift field for a fixed number of steps.

    Stage 5 should call this (or ``model.transport``) as the canonical transport map.
    """
    x = np.asarray(expr, dtype=float).copy()
    ns = int(model.default_n_steps if n_steps is None else n_steps)
    ss = model.reverse_step_size if step_size is None else float(step_size)
    c = np.asarray(context, dtype=float)
    if x.shape != c.shape:
        raise ValueError(f"expr/context shape mismatch: {x.shape} vs {c.shape}")
    den = max(1, ns - 1)
    for i in range(max(1, ns)):
        t_frac = float(i / den)
        x = model.reverse_step(x, c, step_size=ss, t=t_frac)
    return x


def transport_states(
    model: TransportModel,
    expr: np.ndarray,
    context: np.ndarray,
    *,
    n_steps: int | None = None,
    step_size: float | None = None,
) -> np.ndarray:
    """Alias for :func:`generate_counterfactuals` (explicit Stage 5 handoff name)."""
    return generate_counterfactuals(model, expr, context, n_steps=n_steps, step_size=step_size)


def per_spot_l2_distance_to_reference(x: np.ndarray, normal_reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(normal_reference, dtype=float)
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    return np.linalg.norm(x - ref, axis=1)


def transport_sanity_metrics(
    expr: np.ndarray,
    transported: np.ndarray,
    labels: np.ndarray,
    normal_reference: np.ndarray,
) -> dict[str, Any]:
    """Before/after sanity summaries including stratum movement and anti-collapse checks."""
    dist_pre = per_spot_l2_distance_to_reference(expr, normal_reference)
    dist_post = per_spot_l2_distance_to_reference(transported, normal_reference)
    delta = dist_pre - dist_post
    movement = np.linalg.norm(np.asarray(transported, dtype=float) - np.asarray(expr, dtype=float), axis=1)
    lab = np.asarray(labels).astype(str)
    out: dict[str, Any] = {
        "all_spots": {
            "mean_dist_pre": float(dist_pre.mean()),
            "mean_dist_post": float(dist_post.mean()),
            "mean_delta_toward_ref": float(delta.mean()),
            "mean_movement_norm": float(movement.mean()),
        }
    }
    for name in ("tumor", "normal", "intermediate"):
        m = lab == name
        if not np.any(m):
            continue
        out[f"label_{name}"] = {
            "n": int(m.sum()),
            "mean_dist_pre": float(dist_pre[m].mean()),
            "mean_dist_post": float(dist_post[m].mean()),
            "mean_delta_toward_ref": float(delta[m].mean()),
            "mean_movement_norm": float(movement[m].mean()),
        }
    tumor_m = lab == "tumor"
    normal_m = lab == "normal"
    if np.any(tumor_m) and np.any(normal_m):
        tumor_move = float(np.mean(movement[tumor_m]))
        normal_move = float(np.mean(movement[normal_m]))
        ratio = tumor_move / max(normal_move, 1e-8)
        out["tumor_vs_normal_movement"] = {
            "mean_tumor_movement_norm": tumor_move,
            "mean_normal_movement_norm": normal_move,
            "tumor_to_normal_movement_ratio": float(ratio),
            "tumor_moves_more_than_normal": bool(tumor_move > normal_move),
        }
    collapse_eps = max(1e-6, float(np.nanmedian(dist_post)) * 0.05)
    out["collapse_guard"] = {
        "distance_std_pre": float(np.std(dist_pre)),
        "distance_std_post": float(np.std(dist_post)),
        "distance_std_ratio_post_over_pre": float(np.std(dist_post) / max(np.std(dist_pre), 1e-8)),
        "fraction_near_reference_post": float(np.mean(dist_post <= collapse_eps)),
        "collapse_epsilon": float(collapse_eps),
    }
    return out


def save_bridge_checkpoint(model: TransportModel, path: str | Path) -> None:
    """Serialize model arrays and training summary (numpy .npz + sidecar JSON)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    backend = getattr(model, "backend", "linear")
    if backend in ("linear", "bayesian_linear"):
        arrays = dict(
            backend=np.array([backend]),
            w_expr=model.w_expr,
            w_ctx=model.w_ctx,
            bias=model.bias,
            alpha=np.array([model.alpha]),
            normal_reference=model.normal_reference,
            reverse_step_size=np.array([model.reverse_step_size]),
            default_n_steps=np.array([model.default_n_steps]),
            ridge_lambda=np.array([model.ridge_lambda]),
        )
        if backend == "bayesian_linear":
            arrays["precision_inv"] = np.asarray(model.precision_inv)
            arrays["noise_var"] = np.array([model.noise_var])
        np.savez_compressed(p, **arrays)
    else:
        try:
            import torch
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError("PyTorch is required to save neural bridge checkpoints.") from ex
        if p.suffix != ".pt":
            p = p.with_suffix(".pt")
        payload = {
            "backend": "neural",
            "state_dict": model.net.state_dict(),
            "alpha": float(model.alpha),
            "normal_reference": np.asarray(model.normal_reference),
            "reverse_step_size": float(model.reverse_step_size),
            "default_n_steps": int(model.default_n_steps),
            "ridge_lambda": float(model.ridge_lambda),
            "device": str(model.device),
            "input_dim": int(model.net[0].in_features) if len(model.net) > 0 else None,
            "output_dim": int(model.net[-1].out_features) if len(model.net) > 0 else None,
            "hidden_dim": int(getattr(model.net[0], "out_features", 0)) if len(model.net) > 0 else 0,
            "depth_linear_layers": int(sum(1 for m in model.net if m.__class__.__name__ == "Linear")),
        }
        torch.save(payload, p)
    meta = p.parent / f"{p.stem}.summary.json"
    meta.write_text(json.dumps(model.training_summary.to_dict(), indent=2), encoding="utf-8")


def load_bridge_checkpoint(path: str | Path) -> TransportModel:
    p = Path(path)
    meta_path = p.parent / f"{p.stem}.summary.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing training summary sidecar: {meta_path}")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    fields = BridgeTrainingSummary.__dataclass_fields__
    summary = BridgeTrainingSummary(**{k: raw[k] for k in fields if k in raw})
    suffix = p.suffix.lower()
    if suffix == ".pt":
        try:
            import torch
            from torch import nn
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError("PyTorch is required to load neural bridge checkpoints.") from ex
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if payload.get("backend") != "neural":
            raise ValueError(f"Unsupported .pt checkpoint backend: {payload.get('backend')}")
        input_dim = int(payload.get("input_dim"))
        output_dim = int(payload.get("output_dim"))
        hidden_dim = int(payload.get("hidden_dim", 128))
        depth_linear_layers = int(payload.get("depth_linear_layers", 2))
        depth_hidden = max(1, depth_linear_layers - 1)
        layers: list[nn.Module] = []
        cur = input_dim
        for _ in range(depth_hidden):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.GELU())
            cur = hidden_dim
        layers.append(nn.Linear(cur, output_dim))
        net = nn.Sequential(*layers)
        net.load_state_dict(payload["state_dict"])
        return NeuralTransportModel(
            net=net,
            alpha=float(payload["alpha"]),
            normal_reference=np.asarray(payload["normal_reference"]),
            reverse_step_size=float(payload["reverse_step_size"]),
            default_n_steps=int(payload["default_n_steps"]),
            ridge_lambda=float(payload["ridge_lambda"]),
            training_summary=summary,
            device="cpu",
        )
    data = np.load(p, allow_pickle=False)
    saved_backend = "linear"
    if "backend" in data.files:
        raw = np.asarray(data["backend"]).reshape(-1)
        if raw.size > 0:
            saved_backend = str(raw[0])
    common_kwargs = dict(
        w_expr=np.asarray(data["w_expr"]),
        w_ctx=np.asarray(data["w_ctx"]),
        bias=np.asarray(data["bias"]),
        alpha=float(np.asarray(data["alpha"]).reshape(-1)[0]),
        normal_reference=np.asarray(data["normal_reference"]),
        reverse_step_size=float(np.asarray(data["reverse_step_size"]).reshape(-1)[0]),
        default_n_steps=int(np.asarray(data["default_n_steps"]).reshape(-1)[0]),
        ridge_lambda=float(np.asarray(data["ridge_lambda"]).reshape(-1)[0]),
        training_summary=summary,
    )
    if saved_backend == "bayesian_linear" and "precision_inv" in data.files and "noise_var" in data.files:
        return BayesianRidgeBridgeModel(
            **common_kwargs,
            precision_inv=np.asarray(data["precision_inv"]),
            noise_var=float(np.asarray(data["noise_var"]).reshape(-1)[0]),
        )
    return SpatialBridgeModel(**common_kwargs)


def stage_5_handoff_contract() -> dict[str, Any]:
    """Static documentation for downstream perturbation extraction (Stage 5)."""
    return {
        "entry_points": {
            "transport": "omega_spatial.model.generate_counterfactuals(model, expr, context, n_steps=..., step_size=...)",
            "transport_alias": "omega_spatial.model.transport_states(...)",
            "method": "SpatialBridgeModel.transport(expr, context, n_steps=..., step_size=...)",
        },
        "input_ordering": [
            "expr: float array, shape (n_spots, n_genes), same row order as bundle.obs and Stage 3 context.",
            "context: float array, shape (n_spots, n_genes), neighbor-mean expression from Stage 3.",
            "labels: only required during training; inference uses the fitted normal_reference inside the model.",
        ],
        "outputs_guaranteed": [
            "transported expression float array, shape (n_spots, n_genes), finite values if inputs are finite.",
            "perturbation vector per spot: transported - expr (computed by caller).",
        ],
        "diagnostics_stage5_should_preserve": [
            "transport_sanity_metrics(expr, transported, labels, model.normal_reference)",
            "training_summary (JSON sidecar next to checkpoint)",
            "random seed used for any stochastic validation (UMAP); model fit is deterministic given data and config.",
        ],
    }
