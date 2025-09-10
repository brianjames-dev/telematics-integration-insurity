import numpy as np, pandas as pd
from src.ml.glm import PoissonGLM

def test_poisson_glm_basic():
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n,2))
    beta_true = np.array([ -2.0, 0.8, 0.5 ])  # intercept + 2 features
    off = np.zeros(n)
    eta = beta_true[0] + X @ beta_true[1:] + off
    mu = np.exp(eta)
    y = rng.poisson(mu)
    model = PoissonGLM.fit(X, y, off, feature_names=["x1","x2"])
    mu_hat = model.predict_mu(X, off)
    # sanity: mean absolute error on mu is bounded
    assert np.mean(np.abs(mu_hat - mu)) < 0.5
