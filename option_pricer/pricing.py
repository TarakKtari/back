import numpy as np
from scipy.stats import norm

def forward_delta_to_strike(delta_f, opt_type, F, tau, sigma):
    delta_f = np.asarray(delta_f)
    opt_type = np.asarray(opt_type).astype(str)
    F = np.asarray(F)
    tau = np.asarray(tau)
    sigma = np.asarray(sigma)
    chi = np.where(np.char.startswith(np.char.lower(opt_type), "c"), 1.0, -1.0)
    d1 = chi * norm.ppf(chi * delta_f)
    K = F * np.exp(-d1 * sigma * np.sqrt(tau) + 0.5 * sigma ** 2 * tau)
    return K

def garman_kohlhagen(option_type, S, K, tau, rd, rf, sigma):
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if option_type.lower() == "call":
        price = S * np.exp(-rf * tau) * norm.cdf(d1) - K * np.exp(-rd * tau) * norm.cdf(d2)
    else:
        price = K * np.exp(-rd * tau) * norm.cdf(-d2) - S * np.exp(-rf * tau) * norm.cdf(-d1)
    return price
