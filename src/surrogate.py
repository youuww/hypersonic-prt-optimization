"""Gaussian Process surrogate for turbulent Prandtl number prediction.

This module wraps BoTorch's SingleTaskGP into a domain-specific interface
for the Pr_t calibration problem.  It handles:

  - Automatic normalization of inputs X to [0, 1] and standardization of Y
  - Training (MLL optimization) via BoTorch's fit_gpytorch_mll
  - Prediction with uncertainty quantification in physical units
  - Model persistence (save / load checkpoints)

The GP maps a 3D feature vector [Mach, Tw/Taw, theta_pg] to the optimal
turbulent Prandtl number (scalar output).

Usage:
    from surrogate import PrtSurrogate
    import torch

    # Suppose we have 5 calibration runs
    X = torch.tensor([
        [13.6, 0.186, 0.0],
        [ 8.0, 0.30,  0.0],
        ...
    ])
    Y = torch.tensor([[0.566], [0.72], ...])

    model = PrtSurrogate(train_X=X, train_Y=Y)
    mean, std = model.predict(X_test)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Physical bounds for the input features                              #
# ------------------------------------------------------------------ #
#  These define the "design space" over which the GP is valid.
#  Used for input normalization  (X_norm = (X - lower) / (upper - lower)).
#
#       Feature 0: Mach number
#       Feature 1: Wall temperature ratio  Tw / Taw
#       Feature 2: Pressure-gradient angle [deg]

DEFAULT_BOUNDS = torch.tensor(
    [[2.5,   0.10,  0.0],      # lower bounds
     [14.0,  1.00, 24.0]],     # upper bounds
    dtype=torch.float64,
)


@dataclass
class PredictionResult:
    """Container for GP prediction output in physical units."""
    mean: Tensor       # predicted Pr_t  (shape [n])
    std: Tensor        # predictive standard deviation  (shape [n])

    @property
    def lower(self) -> Tensor:
        """Mean - 2*std  (~ 95 % CI lower bound)."""
        return self.mean - 2.0 * self.std

    @property
    def upper(self) -> Tensor:
        """Mean + 2*std  (~ 95 % CI upper bound)."""
        return self.mean + 2.0 * self.std


class PrtSurrogate:
    """GP surrogate:  [Mach, Tw/Taw, theta_pg]  -->  optimal Pr_t.

    Parameters
    ----------
    train_X : Tensor, shape (n, 3)
        Training inputs in *physical* units.
    train_Y : Tensor, shape (n, 1)
        Observed optimal Pr_t values.
    bounds  : Tensor, shape (2, 3), optional
        [lower; upper] bounds for input normalization.
        Defaults to the thesis design space (Mach 2.5-14, etc.).
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        bounds: Optional[Tensor] = None,
    ) -> None:
        if train_X.ndim != 2 or train_X.shape[1] != 3:
            raise ValueError(
                f"train_X must have shape (n, 3), got {train_X.shape}"
            )
        if train_Y.ndim != 2 or train_Y.shape[1] != 1:
            raise ValueError(
                f"train_Y must have shape (n, 1), got {train_Y.shape}"
            )

        self.bounds = (bounds if bounds is not None else DEFAULT_BOUNDS).to(
            dtype=torch.float64
        )
        self._train_X = train_X.to(dtype=torch.float64)
        self._train_Y = train_Y.to(dtype=torch.float64)

        self.model: SingleTaskGP = self._build_and_fit()

    # ------------------------------------------------------------------ #
    #                     Model construction                               #
    # ------------------------------------------------------------------ #
    def _build_and_fit(self) -> SingleTaskGP:
        """Constructs a SingleTaskGP with Matern-5/2 kernel and fits it.

        BoTorch's Normalize and Standardize transforms handle all
        scaling internally - we pass raw physical-unit data.
        """
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=3,   # one lengthscale per input dimension
            )
        )

        model = SingleTaskGP(
            train_X=self._train_X,
            train_Y=self._train_Y,
            covar_module=covar_module,
            input_transform=Normalize(d=3, bounds=self.bounds),
            outcome_transform=Standardize(m=1),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        n = self._train_X.shape[0]
        logger.info(
            "GP fitted on %d points | kernel: Matern-5/2 ARD", n
        )
        return model

    # ------------------------------------------------------------------ #
    #                        Prediction                                    #
    # ------------------------------------------------------------------ #
    def predict(self, X_test: Tensor) -> PredictionResult:
        """Predict Pr_t and uncertainty at new points.

        Parameters
        ----------
        X_test : Tensor, shape (n, 3)
            Query points in *physical* units.

        Returns
        -------
        PredictionResult with .mean and .std in physical Pr_t units.
        """
        X_test = X_test.to(dtype=torch.float64)
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model.posterior(X_test)
            mean = posterior.mean.squeeze(-1)       # (n,)
            std = posterior.variance.squeeze(-1).sqrt()  # (n,)
        return PredictionResult(mean=mean, std=std)

    # ------------------------------------------------------------------ #
    #                     Online update                                    #
    # ------------------------------------------------------------------ #
    def update(self, X_new: Tensor, Y_new: Tensor) -> None:
        """Append new observations and retrain the GP from scratch.

        For small datasets (< 100 points), full retraining is fast
        and avoids error accumulation from incremental updates.
        """
        if X_new.ndim == 1:
            X_new = X_new.unsqueeze(0)
        if Y_new.ndim == 1:
            Y_new = Y_new.unsqueeze(0)

        self._train_X = torch.cat([self._train_X, X_new.to(torch.float64)])
        self._train_Y = torch.cat([self._train_Y, Y_new.to(torch.float64)])
        self.model = self._build_and_fit()

        logger.info(
            "GP updated: %d total training points", self._train_X.shape[0]
        )

    # ------------------------------------------------------------------ #
    #                     Persistence                                      #
    # ------------------------------------------------------------------ #
    def save(self, directory: Path) -> None:
        """Save model state + training data to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), directory / "gp_state.pt")
        torch.save(self._train_X, directory / "train_X.pt")
        torch.save(self._train_Y, directory / "train_Y.pt")
        torch.save(self.bounds, directory / "bounds.pt")

        meta = {
            "n_train": int(self._train_X.shape[0]),
            "input_dim": int(self._train_X.shape[1]),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("GP saved to %s  (%d points)", directory, meta["n_train"])

    @classmethod
    def load(cls, directory: Path) -> PrtSurrogate:
        """Reconstruct a PrtSurrogate from a saved checkpoint."""
        directory = Path(directory)

        train_X = torch.load(directory / "train_X.pt", weights_only=True)
        train_Y = torch.load(directory / "train_Y.pt", weights_only=True)
        bounds = torch.load(directory / "bounds.pt", weights_only=True)

        surrogate = cls(train_X=train_X, train_Y=train_Y, bounds=bounds)

        surrogate.model.load_state_dict(
            torch.load(directory / "gp_state.pt", weights_only=True)
        )
        logger.info(
            "GP loaded from %s  (%d points)",
            directory,
            train_X.shape[0],
        )
        return surrogate

    # ------------------------------------------------------------------ #
    #                     Diagnostics                                      #
    # ------------------------------------------------------------------ #
    @property
    def n_train(self) -> int:
        """Number of training observations."""
        return self._train_X.shape[0]

    @property
    def lengthscales(self) -> Tensor:
        """Fitted ARD lengthscales (one per input dimension)."""
        return (
            self.model.covar_module.base_kernel.lengthscale.detach().squeeze()
        )

    def summary(self) -> str:
        """Human-readable summary of the fitted GP state."""
        ls = self.lengthscales
        return (
            f"  PrtSurrogate  ({self.n_train} training points)\n"
            f"  Kernel:       Matern-5/2  (ARD)\n"
            f"  Lengthscales: Mach={ls[0]:.3f}  Tw/Taw={ls[1]:.3f}"
            f"  theta_pg={ls[2]:.3f}\n"
            f"  Noise var:    "
            f"{self.model.likelihood.noise.item():.4e}"
        )
