import pandas as pd
from src.opg_backlog_sim.emulator import train_emulator, predict_kpi

def main():
    gp = train_emulator([160,180,200],[70,80,90,100],[16,18,20,22],days=60,seed=42)
    rows=[]
    for a in [170,190,210]:
        for f in [75,85,95]:
            for w in [16,18,20]:
                m,s = predict_kpi(gp,a,f,w)
                rows.append(dict(arrivals=a, fte=f, wip=w, kpi_mean=m, kpi_sd=s))
    df = pd.DataFrame(rows).sort_values('kpi_mean')
    print(df.head(10).to_string(index=False))

if __name__=='__main__': main()
