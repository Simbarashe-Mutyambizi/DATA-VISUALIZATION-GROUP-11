import matplotlib.pyplot as plt


def plot_bar(values, labels, title=None, ylabel=None, figsize=(8,4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if title: ax.set_title(title)
    if ylabel: ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig, ax

# For Initial attempt

if __name__ == "__main__":
    vals = [0.195, 0.082, 0.246, 0.083]
    names = ["Unleaded_91", "Diesel", "Premium 95","Premium 98"]
    fig, ax = plot_bar(vals, names, title="R²(test) Comparison between fuel types", ylabel="R²")
    fig.savefig("R2_comparison_amongst_fuel_types.png", dpi=1000) 
    plt.show()
