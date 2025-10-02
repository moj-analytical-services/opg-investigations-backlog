import streamlit as st, yaml, simpy, random, pandas as pd
from pathlib import Path
from src.opg_backlog_sim.simulation import BacklogSimulator

st.set_page_config(page_title='OPG Backlog — Scenario Runner', layout='wide')
st.title('OPG Investigations Backlog — Scenario Runner (Transparent)')

cfg_path = st.text_input('Config file (YAML)', 'configs/config.yaml')
scenario = st.selectbox('Scenario', ['plan_a','plan_b'])

if st.button('Run simulation'):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding='utf-8'))
    env = simpy.Environment()
    sim = BacklogSimulator(env,
        arrivals_per_week=cfg['arrivals']['weekly_total'],
        mix=cfg['arrivals']['mix'],
        io_fte=cfg['staffing']['investigators_fte'],
        wip_limit=cfg['policies']['wip_limit_per_investigator'],
        seed=cfg.get('seed',42))
    metrics=[]
    def kpis():
        while True:
            day = int(env.now)
            open_cases = [c for c in sim.cases if c.state!='CLOSED']
            ages = [day - c.opened_day for c in open_cases]
            metrics.append(dict(day=day, open=len(open_cases), over30=sum(a>30 for a in ages), over60=sum(a>60 for a in ages), over120=sum(a>120 for a in ages)))
            yield env.timeout(5)
    env.process(sim.arrival_process())
    env.process(kpis())
    env.run(until=cfg.get('horizon_days',120))
    kpi = pd.DataFrame(metrics)
    st.subheader('Backlog size (open cases)'); st.line_chart(kpi.set_index('day')['open'])
    st.subheader('Ageing profile'); st.line_chart(kpi.set_index('day')[['over30','over60','over120']])
    st.download_button('Download KPI CSV', kpi.to_csv(index=False), 'kpi.csv', 'text/csv')
