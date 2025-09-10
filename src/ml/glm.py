from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# --------- Utilities ---------
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])

# --------- Poisson GLM with log link + offset (IRLS) ---------
@dataclass
class PoissonGLM:
    feature_names: List[str]
    beta: np.ndarray
    l2: float = 1e-6
    max_iter: int = 50
    tol: float = 1e-6

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray, offset: np.ndarray, feature_names: List[str], l2: float = 1e-6, max_iter: int = 50, tol: float = 1e-6) -> "PoissonGLM":
        X1 = add_intercept(X)   # include intercept
        beta = np.zeros(X1.shape[1])
        for _ in range(max_iter):
            eta = X1 @ beta + offset
            mu = np.exp(eta)
            W = mu  # weights
            z = eta + (y - mu) / np.maximum(mu, 1e-9)
            # solve (X^T W X + l2 I) beta = X^T W (z - offset)
            WX = X1 * W[:, None]
            A = X1.T @ WX + l2 * np.eye(X1.shape[1])
            b = X1.T @ (W * (z - offset))
            beta_new = np.linalg.solve(A, b)
            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new
        return PoissonGLM(feature_names=["intercept"] + feature_names, beta=beta, l2=l2, max_iter=max_iter, tol=tol)

    def predict_mu(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        X1 = add_intercept(X)
        return np.exp(X1 @ self.beta + offset)

    def to_json(self) -> Dict:
        return {"type": "poisson_glm", "feature_names": self.feature_names, "beta": self.beta.tolist()}

    @staticmethod
    def from_json(obj: Dict) -> "PoissonGLM":
        return PoissonGLM(feature_names=obj["feature_names"], beta=np.array(obj["beta"]))


# --------- Lognormal regression (on log(y)) with ridge ---------
@dataclass
class LognormalReg:
    feature_names: List[str]
    beta: np.ndarray           # includes intercept at index 0
    sigma2: float              # variance of residuals on log-scale

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray, feature_names: List[str],
            l2: float = 1.0, max_mu_clip: float = 12.0) -> "LognormalReg":
        """
        Regress log(y) on [1, X] with L2 ridge.
        - Drops non-positive y safely.
        - Does not penalize intercept.
        - Clips sigma2 to [0.01, 1.5] for stability.
        """
        mask = y > 0
        X = X[mask]
        y = y[mask]
        ylog = np.log(y + 1e-8)
        lo, hi = np.quantile(ylog, [0.01, 0.99])
        ylog = np.clip(ylog, lo, hi)

        n, p = X.shape
        X1 = np.hstack([np.ones((n, 1)), X])
        I = np.eye(p + 1)
        I[0, 0] = 0.0  # no penalty on intercept

        A = X1.T @ X1 + l2 * I
        b = X1.T @ ylog
        beta = np.linalg.solve(A, b)

        resid = ylog - (X1 @ beta)
        # ddof guards; clip to reasonable range to avoid exp overflow
        dof = max(1, n - (p + 1))
        sigma2 = float(np.var(resid, ddof=dof))
        sigma2 = float(np.clip(sigma2, 0.01, 1.5))

        return LognormalReg(feature_names=["intercept"] + feature_names, beta=beta, sigma2=sigma2)

    def predict_mean(self, X: np.ndarray, max_mu_clip: float = 6.0) -> np.ndarray:
        """
        E[Y | X] for lognormal = exp(mu + 0.5*sigma2) with
        mu = [1, X] @ beta; clip mu to avoid huge exponentials.
        max_mu_clip=12 -> caps mean around exp(12) ~ 162k (ample for synthetic).
        """
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])
        mu = X1 @ self.beta
        mu = np.clip(mu, -2.0, max_mu_clip)
        return np.exp(mu + 0.5 * self.sigma2)

    def to_json(self) -> dict:
        return {"type": "lognormal_reg", "feature_names": self.feature_names,
                "beta": self.beta.tolist(), "sigma2": self.sigma2}

    @staticmethod
    def from_json(obj: dict) -> "LognormalReg":
        return LognormalReg(feature_names=obj["feature_names"],
                            beta=np.array(obj["beta"]), sigma2=float(obj["sigma2"]))



# --------- Logistic GLM with logit link + offset (IRLS) ---------
@dataclass
class LogisticGLM:
    feature_names: List[str]
    beta: np.ndarray
    l2: float = 1e-6
    max_iter: int = 100
    tol: float = 1e-6

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray, offset: np.ndarray, feature_names: List[str],
            l2: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> "LogisticGLM":
        """
        IRLS with ridge penalty; numerically stabilized:
          - z-scoring should be done by caller (train_glm)
          - clip W to [eps, 0.25] to avoid near-singular normal equations
        """
        def add_intercept(A): return np.hstack([np.ones((A.shape[0], 1)), A])
        X1 = add_intercept(X)
        beta = np.zeros(X1.shape[1], dtype=float)
        eps = 1e-5  # was 1e-9

        I = np.eye(X1.shape[1])
        for _ in range(max_iter):
            eta = X1 @ beta + offset
            # stable sigmoid
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
            W = np.clip(p * (1.0 - p), eps, 0.25)  # cap at 0.25 (max of p*(1-p))
            z = eta + (y - p) / W

            WX = X1 * W[:, None]
            A = X1.T @ WX + l2 * I
            b = X1.T @ (W * (z - offset))
            beta_new = np.linalg.solve(A, b)

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        return LogisticGLM(feature_names=["intercept"] + feature_names, beta=beta, l2=l2, max_iter=max_iter, tol=tol)

    def predict_prob(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        def add_intercept(A): return np.hstack([np.ones((A.shape[0], 1)), A])
        X1 = add_intercept(X)
        eta = X1 @ self.beta + offset
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))

    def to_json(self) -> dict:
        return {"type": "logistic_glm", "feature_names": self.feature_names, "beta": self.beta.tolist()}

    @staticmethod
    def from_json(obj: dict) -> "LogisticGLM":
        return LogisticGLM(feature_names=obj["feature_names"], beta=np.array(obj["beta"]))
