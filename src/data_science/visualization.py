# visualization.py
import matplotlib.pyplot as plt


def correlation_heatmap(corr, labels=None):
    fig, ax = plt.subplots()
    cax = ax.imshow(corr, aspect="auto")
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(cax, ax=ax)
    fig.tight_layout()
    return fig, ax


def boxplot_groups(data, labels=None):
    fig, ax = plt.subplots()
    ax.boxplot(data, vert=True)
    if labels:
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Box-and-Whisker Plot")
    fig.tight_layout()
    return fig, ax
