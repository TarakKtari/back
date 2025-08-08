import numpy as np

def ssvi_total_variance(k, theta, rho, eta, gamma):
    phi = eta * theta ** (-gamma)
    term = phi * k + rho
    return 0.5 * theta * (1 + rho * phi * k + np.sqrt(term ** 2 + 1 - rho ** 2))
