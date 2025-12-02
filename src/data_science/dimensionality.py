# dimensionality.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_fit_transform(X, n_components=2, scale=True):
    if scale:
        X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)
    return pca, Z


def plot_explained_variance(pca):
    fig, ax = plt.subplots()
    ax.plot(
        np.arange(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
    )
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA Explained Variance")
    fig.tight_layout()
    return fig, ax
