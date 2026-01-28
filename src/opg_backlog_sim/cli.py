import argparse
import simpy
from .simulation import BacklogSimulator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--arrivals", type=int, default=180)
    p.add_argument("--fte", type=int, default=85)
    p.add_argument("--wip", type=int, default=20)
    a = p.parse_args()
    env = simpy.Environment()
    sim = BacklogSimulator(
        env, arrivals_per_week=a.arrivals, io_fte=a.fte, wip_limit=a.wip
    )
    sim.run(days=a.days)
    print("Done.")
