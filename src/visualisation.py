import matplotlib.pyplot as plt


def plot_daily_backlog(daily):
    ax = daily.plot(x="date", y="backlog", figsize=(10, 4), title="Daily backlog")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cases")
    return ax.get_figure()


def plot_forecast(hist, fc):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist["date"], hist["backlog"], label="History")
    ax.plot(fc["date"], fc["pred_backlog"], label="Forecast")
    ax.set_title("Backlog Forecast")
    ax.legend()
    return fig
