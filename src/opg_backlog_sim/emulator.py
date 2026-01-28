from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
import numpy as np
import simpy
import random
from .simulation import BacklogSimulator


def run_des_once(arrivals, fte, wip, days=60, seed=42):
    rnd = random.Random(seed)
    env = simpy.Environment()
    sim = BacklogSimulator(
        env, arrivals_per_week=arrivals, io_fte=fte, wip_limit=wip, seed=seed
    )
    metrics = []

    def kpis():
        while True:
            day = int(env.now)
            open_cases = [c for c in sim.cases if c.state != "CLOSED"]
            metrics.append((day, len(open_cases)))
            yield env.timeout(5)

    env.process(sim.arrival_process())
    env.process(kpis())
    env.run(until=days)
    return float(np.mean([m[1] for m in metrics])) if metrics else 0.0


def train_emulator(grid_arrivals, grid_fte, grid_wip, days=60, seed=42):
    X = []
    y = []
    for a in grid_arrivals:
        for f in grid_fte:
            for w in grid_wip:
                X.append([a, f, w])
                y.append(run_des_once(a, f, w, days=days, seed=seed))
    X = np.array(X)
    y = np.array(y)
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[40, 10, 3], nu=1.5
    ) + WhiteKernel(1.0)
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=seed
    )
    gp.fit(X, y)
    return gp


def predict_kpi(gp, arrivals, fte, wip):
    x = np.array([[arrivals, fte, wip]])
    m, s = gp.predict(x, return_std=True)
    return float(m), float(s)
