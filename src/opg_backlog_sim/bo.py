import numpy as np
from scipy.stats import norm

def expected_improvement(mu, sigma, best):
    sigma=np.maximum(sigma,1e-6); z=(best-mu)/sigma; return (best-mu)*norm.cdf(z)+sigma*norm.pdf(z)

def suggest_next(gp, bounds, n_candidates=2000, best_so_far=np.inf, constraints=None):
    cand=np.random.rand(n_candidates,3); keys=list(bounds.keys())
    for i,k in enumerate(keys): lo,hi=bounds[k]; cand[:,i]=lo+cand[:,i]*(hi-lo)
    mu,std=gp.predict(cand,return_std=True); ei=expected_improvement(mu,std,best_so_far)
    mask=np.ones(len(cand),dtype=bool)
    if constraints:
        for fn in constraints: mask &= fn(cand)
    ei[~mask]=-np.inf; idx=int(np.argmax(ei)); return cand[idx], mu[idx], std[idx], ei[idx]
