from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig


@dataclass
class SpatialBridgeModel:
    """Lightweight conditional score proxy for spatial Schrödinger bridge."""

    w_expr: np.ndarray
    w_ctx: np.ndarray
    bias: np.ndarray
    alpha: float

    def score(self, x_t: np.ndarray, context: np.ndarray) -> np.ndarray:
        return x_t @ self.w_expr + context @ self.w_ctx + self.bias

    def reverse_step(self, x_t: np.ndarray, context: np.ndarray, step_size: float = 0.2) -> np.ndarray:
        return x_t + step_size * self.score(x_t, context) + self.alpha * (context - x_t)


def train_spatial_bridge(
    expr: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    cfg: PipelineConfig,
) -> SpatialBridgeModel:
    dim = expr.shape[1]
    normal_idx = np.where(labels == "normal")[0]
    tumor_idx = np.where(labels == "tumor")[0]
    if len(normal_idx) == 0:
        normal_idx = np.arange(expr.shape[0])
    if len(tumor_idx) == 0:
        tumor_idx = np.arange(expr.shape[0])

    normal_mu = expr[normal_idx].mean(axis=0, keepdims=True)
    target_drift = normal_mu - expr

    # Closed-form ridge fit: [X, C, 1] -> target_drift.
    z = np.concatenate([expr, context, np.ones((expr.shape[0], 1))], axis=1)
    lam = 1e-3
    a = z.T @ z + lam * np.eye(z.shape[1])
    b = z.T @ target_drift
    w = np.linalg.solve(a, b)

    w_expr = w[:dim]
    w_ctx = w[dim : 2 * dim]
    bias = w[-1]
    return SpatialBridgeModel(
        w_expr=w_expr,
        w_ctx=w_ctx,
        bias=bias,
        alpha=cfg.spatial.spatial_smoothing_alpha,
    )


def generate_counterfactuals(
    model: SpatialBridgeModel,
    expr: np.ndarray,
    context: np.ndarray,
    n_steps: int = 8,
) -> np.ndarray:
    x = expr.copy()
    for _ in range(n_steps):
        x = model.reverse_step(x, context, step_size=0.2)
    return x
