import simpy, random
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Case:
    id:int; segment:str; opened_day:int; state:str='TRIAGE'

@dataclass
class StaffPool:
    env: simpy.Environment; io_capacity:int; wip_limit:int; resource: simpy.PriorityResource=field(init=False)
    def __post_init__(self): self.resource = simpy.PriorityResource(self.env, capacity=self.io_capacity)

class BacklogSimulator:
    def __init__(self, env, arrivals_per_week=180, mix=None, io_fte=85, wip_limit=20, seed=42):
        self.env=env; self.r=random.Random(seed); self.arrivals=arrivals_per_week; self.mix=mix or {'deputyship':0.35,'lpa_finance':0.45,'lpa_health':0.20}
        self.staff=StaffPool(env, io_capacity=max(1, io_fte//5), wip_limit=wip_limit)
        self.cases=[]; self.completed=[]; self.ct=0
    def arrival_process(self):
        while True:
            daily=max(0,int(self.arrivals/5))
            for _ in range(daily):
                seg=self.r.choices(list(self.mix.keys()),weights=list(self.mix.values()))[0]
                c=Case(self.ct, seg, int(self.env.now)); self.ct+=1; self.cases.append(c); self.env.process(self.process_case(c))
            yield self.env.timeout(1)
    def process_case(self, c):
        yield self.env.timeout(self.r.randint(0,2)); c.state='ALLOCATED'
        with self.staff.resource.request(priority=0) as req:
            yield req; yield self.env.timeout(self.r.randint(1,5))
        if self.r.random()<0.2:
            c.state='AWAITING_EXTERNAL'; yield self.env.timeout(self.r.randint(3,15))
        with self.staff.resource.request(priority=0) as req2:
            yield req2; yield self.env.timeout(self.r.randint(1,5))
        r=self.r.random()
        if r<0.25: c.state='CLOSED'
        elif r<0.55:
            c.state='COP_PREP'; yield self.env.timeout(self.r.randint(2,10)); c.state='COP_SUBMITTED'; yield self.env.timeout(self.r.randint(7,21)); c.state='CLOSED'
        else: c.state='CLOSED'
        self.completed.append(c)
    def run(self, days=30): self.env.process(self.arrival_process()); self.env.run(until=days)
