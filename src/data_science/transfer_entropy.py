# transfer_entropy.py
# Simple discrete transfer entropy estimator using histogram binning.
# This is a coarse estimator; for serious work consider specialised packages.
import numpy as np


def _hist_prob(*arrays, bins=8):
    hist, edges = np.histogramdd(np.column_stack(arrays), bins=bins)
    p = hist / hist.sum()
    return p + 1e-12  # add tiny value for numerical stability


def transfer_entropy(x, y, k=1, l=1, bins=8):
    """
    Estimate TE X->Y using discrete binning.
    x, y: 1D arrays; k: history length of Y; l: history length of X.
    Returns scalar TE (nats).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = max(k, l)
    Yf = y[n:]  # future Y
    Yp = (
        np.column_stack([y[n - i - 1 : -i - 1] for i in range(k)])
        if k > 0
        else np.empty((len(Yf), 0))
    )
    Xp = (
        np.column_stack([x[n - j - 1 : -j - 1] for j in range(l)])
        if l > 0
        else np.empty((len(Yf), 0))
    )

    # Joint probabilities
    p_yf_yp_xp = _hist_prob(Yf, Yp, Xp, bins=bins)
    p_yf_yp = _hist_prob(Yf, Yp, bins=bins)
    p_yf_yp_xp = p_yf_yp_xp[p_yf_yp_xp > 0]
    p_yf_yp = p_yf_yp[p_yf_yp > 0]

    # Marginals required for conditional entropies:
    # H(Yf | Yp, Xp) - H(Yf | Yp)
    te = np.sum(p_yf_yp_xp * np.log(p_yf_yp_xp)) - np.sum(p_yf_yp * np.log(p_yf_yp))
    return -te
