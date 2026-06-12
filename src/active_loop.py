"""Active Learning loop for Pr_t calibration.

Implements Bayesian Optimization to decide WHERE in the design space
[Mach, Tw/Taw, theta_pg] the next expensive CFD simulation should be run.

Instead of a naive grid search (which would require hundreds of SU2 runs),
this module uses the GP surrogate's uncertainty to focus computational
budget on the regions that matter most.

Architecture
------------
    ActiveCalibrationLoop
        |
        +--> PrtSurrogate   (GP model — predicts Pr_t + uncertainty)
        +--> SU2Interface    (CFD solver — runs SU2 at a given point)
        +--> Acquisition     (BoTorch — decides next query point)

Workflow (one step):
    1. Acquisition function proposes the next X* (highest expected value)
    2. Optimizer within SU2 finds optimal Pr_t at X*  (Brent's method)
    3. GP is updated with the new (X*, Pr_t*) pair
    4. Repeat until budget is exhausted or uncertainty is low everywhere

Usage:
    from active_loop import ActiveCalibrationLoop
    loop = ActiveCalibrationLoop.from_initial_data(X_init, Y_init)
    for _ in range(budget):
        result = loop.step()
        print(result)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from botorch.acquisition import (
    UpperConfidenceBound,
    qExpectedImprovement,
)
from botorch.optim import optimize_acqf

from surrogate import PrtSurrogate, DEFAULT_BOUNDS

logger = logging.getLogger(__name__)


class Decision(Enum):
    """Outcome of evaluating a candidate point."""
    CONFIDENT = "confident"     # GP uncertainty below threshold
    NEEDS_CFD = "needs_cfd"     # GP uncertainty too high — run SU2


@dataclass
class StepResult:
    """Record of one active-learning iteration."""
    iteration: int
    x_candidate: list[float]       # [Mach, Tw/Taw, theta_pg]
    decision: Decision          # Whether we trusted the GP or ran CFD
    predicted_prt: float        # GP mean prediction at x_candidate (physical units)
    predicted_std: float        # GP std dev at x_candidate (physical units)
    observed_prt: Optional[float] = None    # filled after CFD
    observed_rmse: Optional[float] = None   # filled after CFD
    wall_time_sec: float = 0.0      # time taken for this iteration (including CFD if run)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "Mach": self.x_candidate[0],
            "Tw_Taw": self.x_candidate[1],
            "theta_pg": self.x_candidate[2],
            "decision": self.decision.value,
            "predicted_prt": self.predicted_prt,
            "predicted_std": self.predicted_std,
            "observed_prt": self.observed_prt,
            "observed_rmse": self.observed_rmse,
            "wall_time_sec": self.wall_time_sec,
        }


class ActiveCalibrationLoop:
    """Bayesian Optimization loop for Pr_t calibration.

    Parameters
    ----------
    surrogate : PrtSurrogate
        Fitted GP model.
    bounds : Tensor, shape (2, 3)
        Design space bounds for acquisition optimization.
    uncertainty_threshold : float
        If GP std < threshold at the candidate point, we skip CFD and
        trust the GP prediction (in physical Pr_t units).
    beta : float
        Exploration-exploitation trade-off for Upper Confidence Bound.
        Higher beta = more exploration.
    checkpoint_dir : Path
        Where to save iteration logs and model checkpoints.
    """

    def __init__(
        self,
        surrogate: PrtSurrogate,
        bounds: Optional[Tensor] = None,
        uncertainty_threshold: float = 0.03,
        beta: float = 2.0,
        checkpoint_dir: Path = Path("checkpoints"),
    ) -> None:
        self.surrogate = surrogate
        self.bounds = (
            bounds if bounds is not None else DEFAULT_BOUNDS
        ).to(torch.float64)
        self.uncertainty_threshold = uncertainty_threshold
        self.beta = beta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[StepResult] = []
        self._iteration = 0

    # ------------------------------------------------------------------ #
    #          Factory: create from initial data                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_initial_data(
        cls,
        train_X: Tensor,
        train_Y: Tensor,
        **kwargs,
    ) -> ActiveCalibrationLoop:
        """Build surrogate from raw data and create the loop."""
        surrogate = PrtSurrogate(train_X=train_X, train_Y=train_Y)
        return cls(surrogate=surrogate, **kwargs)

    # ------------------------------------------------------------------ #
    #        Core: propose the next point to evaluate                      #
    # ------------------------------------------------------------------ #
    def suggest_next_point(self) -> Tensor:
        """Use Upper Confidence Bound to find the most valuable next query.

        UCB balances:
          - Exploitation  (sample where predicted Pr_t is uncertain)
          - Exploration   (sample where we have no data)

        We MINIMIZE predicted Pr_t error, so we negate UCB to find
        the point with highest uncertainty (= highest information gain).

        Returns
        -------
        Tensor, shape (1, 3) — the proposed [Mach, Tw/Taw, theta_pg].
        """
        acqf = UpperConfidenceBound(
            model=self.surrogate.model,
            beta=self.beta,
        )

        candidate, acq_value = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )
        logger.info(
            "Acquisition suggests X = %s  (acq_value = %.4f)",
            candidate.squeeze().tolist(),
            acq_value.item(),
        )
        return candidate

    # ------------------------------------------------------------------ #
    #        Decision logic: trust GP or run CFD?                          #
    # ------------------------------------------------------------------ #
    def evaluate_point(self, x: Tensor) -> tuple[Decision, float, float]:
        """Predict Pr_t at x and decide whether CFD is needed.

        Returns (decision, predicted_mean, predicted_std).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        result = self.surrogate.predict(x)
        mean_val = result.mean.item()
        std_val = result.std.item()

        if std_val < self.uncertainty_threshold:
            decision = Decision.CONFIDENT
        else:
            decision = Decision.NEEDS_CFD

        logger.info(
            "X = %s | Pr_t = %.4f +/- %.4f | Decision: %s",
            x.squeeze().tolist(),
            mean_val,
            std_val,
            decision.value,
        )
        return decision, mean_val, std_val

    # ------------------------------------------------------------------ #
    #        One active-learning iteration                                 #
    # ------------------------------------------------------------------ #
    def step(
        self,
        run_cfd_callback=None,
    ) -> StepResult:
        """Execute one iteration of the active learning loop.

        Parameters
        ----------
        run_cfd_callback : callable, optional
            Function with signature  (x: Tensor) -> (optimal_prt, rmse).
            Called when the GP is uncertain and needs a real CFD run.
            If None and CFD is needed, the step records NEEDS_CFD but
            does not run anything (dry-run mode).

        Returns
        -------
        StepResult with all iteration metadata.
        """
        self._iteration += 1
        t_start = time.time()

        # 1. Propose candidate
        x_candidate = self.suggest_next_point()
        x_list = x_candidate.squeeze().tolist()

        # 2. Evaluate GP prediction + decision
        decision, pred_mean, pred_std = self.evaluate_point(x_candidate)

        result = StepResult(
            iteration=self._iteration,
            x_candidate=x_list,
            decision=decision,
            predicted_prt=pred_mean,
            predicted_std=pred_std,
        )

        # 3. If uncertain and callback available — run CFD
        if decision == Decision.NEEDS_CFD and run_cfd_callback is not None:
            logger.info("Triggering CFD at X = %s ...", x_list)
            observed_prt, observed_rmse = run_cfd_callback(x_candidate)

            result.observed_prt = observed_prt
            result.observed_rmse = observed_rmse

            # 4. Update GP with the new observation
            Y_new = torch.tensor([[observed_prt]], dtype=torch.float64)
            self.surrogate.update(x_candidate, Y_new)

        result.wall_time_sec = time.time() - t_start
        self.history.append(result)

        # 5. Checkpoint
        self._save_checkpoint()

        return result

    # ------------------------------------------------------------------ #
    #                     Persistence                                      #
    # ------------------------------------------------------------------ #
    def _save_checkpoint(self) -> None:
        """Save iteration history and GP state after each step."""
        history_data = [r.to_dict() for r in self.history]
        log_path = self.checkpoint_dir / "active_learning_log.json"
        log_path.write_text(json.dumps(history_data, indent=2))

        self.surrogate.save(self.checkpoint_dir / "gp_model")

    # ------------------------------------------------------------------ #
    #                     Status reporting                                 #
    # ------------------------------------------------------------------ #
    def status(self) -> str:
        """Human-readable status summary."""
        n_cfd = sum(
            1 for r in self.history if r.decision == Decision.NEEDS_CFD
        )
        n_skip = sum(
            1 for r in self.history if r.decision == Decision.CONFIDENT
        )
        lines = [
            f"  Active Learning Status",
            f"  ----------------------",
            f"  Iterations:   {self._iteration}",
            f"  CFD runs:     {n_cfd}",
            f"  GP-only:      {n_skip}",
            f"  GP training:  {self.surrogate.n_train} points",
            f"  Uncertainty:  threshold = {self.uncertainty_threshold}",
        ]
        return "\n".join(lines)
