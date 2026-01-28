import simpy
from src.opg_backlog_sim.simulation import BacklogSimulator


def test_simulation_runs():
    env = simpy.Environment()
    sim = BacklogSimulator(env, arrivals_per_week=20, io_fte=10, wip_limit=20, seed=1)
    sim.run(days=5)
    assert len(sim.cases) > 0
