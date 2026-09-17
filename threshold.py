from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import norm


def threshold_gmm(errors, q=0.99):
    gmm = GaussianMixture(n_components = 2, random_state = 0)
    errors_reshaped = errors.reshape(-1, 1)
    gmm.fit(errors_reshaped)
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    anomaly_idx = np.argmax(means)
    mean, var = means[anomaly_idx], variances[anomaly_idx]
    std = np.sqrt(var)
    
    threshold = norm.ppf(q, loc=mean, scale = std)
    return np.full_like(errors, threshold)

