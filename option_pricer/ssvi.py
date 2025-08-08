import numpy as np
from dataclasses import dataclass
from scipy.optimize import differential_evolution, minimize
from .ssvi_total_variance import ssvi_total_variance

@dataclass
class SSVIResult:
    rho: float
    eta: float
    gamma: float
    theta_curve: any  # pd.Series

    def theta(self, tau: float) -> float:
        idx = self.theta_curve.index.values.astype(float)
        vals = self.theta_curve.values.astype(float)
        return np.interp(tau, idx, vals)

    def total_variance(self, k, tau) -> np.ndarray:
        return ssvi_total_variance(k, self.theta(tau), self.rho, self.eta, self.gamma)

    def implied_vol(self, k, tau) -> np.ndarray:
        return np.sqrt(self.total_variance(k, tau) / tau)

class SSVICalibrator:
    def __init__(self, popsize=60, maxiter=400, seed=1):
        self.popsize = popsize
        self.maxiter = maxiter
        self.seed = seed
        self._bounds = [(-0.999, 0.999), (1e-3, 25), (0.05, 0.9)]
    @staticmethod
    def _mse(params, k, theta, w_obs):
        rho, eta, gamma = params
        w_model = ssvi_total_variance(k, theta, rho, eta, gamma)
        phi = eta * theta**(-gamma)
        penalty  = 1e6 * np.sum(np.maximum(0, theta*phi*(1+abs(rho)) - 4))
        penalty += 1e6 * np.sum(np.maximum(0, theta*phi**2*(1+abs(rho)) - 4))
        return np.mean((w_model - w_obs)**2) + penalty
    def fit(self, df, theta_curve):
        mask_smile = df["side"] != "ATM"
        k_obs      = df.loc[mask_smile, "k"].values
        w_obs      = df.loc[mask_smile, "w"].values
        theta_obs  = df.loc[mask_smile, "theta_tau"].values
        de = differential_evolution(self._mse,
                                    bounds=self._bounds,
                                    args=(k_obs, theta_obs, w_obs),
                                    popsize=self.popsize,
                                    maxiter=self.maxiter,
                                    polish=False,
                                    seed=self.seed)
        local = minimize(self._mse,
                         x0=de.x,
                         args=(k_obs, theta_obs, w_obs),
                         bounds=self._bounds,
                         method="L-BFGS-B",
                         options={"ftol": 1e-8})
        rho, eta, gamma = local.x
        return SSVIResult(rho=float(rho), eta=float(eta), gamma=float(gamma), theta_curve=theta_curve)
