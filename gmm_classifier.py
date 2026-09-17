
import numpy as np
import joblib


def load_gmm_model(model_path="gmm_model.pkl"):
    """
    Load a previously trained GMM model.
    """
    return joblib.load(model_path)


def classify_errors(gmm, reconstruction_errors):
    """
    Classify reconstruction errors using the trained GMM.

    Returns:
        labels: GMM cluster assigned to each reconstruction error
        probabilities: probability of each error belonging to each cluster
    """

    reconstruction_errors = np.asarray(reconstruction_errors).reshape(-1, 1)

    labels = gmm.predict(reconstruction_errors)
    probabilities = gmm.predict_proba(reconstruction_errors)

    return labels, probabilities


def get_cluster_info(gmm):
    """
    Return the learned characteristics of each GMM cluster.
    """

    cluster_info = []

    for i in range(gmm.n_components):

        mean = gmm.means_[i, 0]
        variance = gmm.covariances_[i, 0, 0]
        std = np.sqrt(variance)
        weight = gmm.weights_[i]

        cluster_info.append({
            "cluster": i,
            "mean": mean,
            "std": std,
            "variance": variance,
            "weight": weight
        })

    return cluster_info