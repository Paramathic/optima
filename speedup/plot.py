import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data
file_path = "results/speedup_results.csv"
data = pd.read_csv(file_path)


def inch_to_pts(inch):
    return inch * 72.27

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def prepare_figure(size_fraction=1.):
    fig_dim = set_size(inch_to_pts(5.5), fraction=size_fraction)
    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 6,
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 6,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 1,
        "ytick.labelsize": 4,
        "figure.figsize": fig_dim,
        'lines.linewidth': 0.4,
        'figure.facecolor': "white"
    }

    plt.rcParams.update(tex_fonts)
    plt.style.use('seaborn-v0_8')


def post_process_figure(ax):
    #Set default plt figure to ax
    plt.sca(ax)

    plt.grid(True, which='major', color='grey', alpha=0.2, linewidth=0.3)
    plt.grid(True, which='minor', color='grey', linestyle='--', alpha=0.1, linewidth=0.1)
    plt.minorticks_on()
    plt.tick_params(which='major', axis="y", direction="in", width=0.5, color='grey')
    plt.tick_params(which='minor', axis="y", direction="in", width=0.3, color='grey')
    plt.tick_params(which='major', axis="x", direction="in", width=0.5, color='grey')
    plt.tick_params(which='minor', axis="x", direction="in", width=0.3, color='grey')

    # Remove the upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('grey')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_color('grey')
    ax.spines['left'].set_linewidth(0.5)


# Define the types of layers
def determine_layer_type(row):
    config = row["Configuration"].split(",")
    d_in, d_out = config[2].split("=")[1], config[3].split("=")[1]
    if d_out == d_in:
        return "Self-Attention"
    elif d_out > d_in:
        return "Up-Projection"
    else:
        return "Down-Projection"

def determine_model(row):
    return row["Configuration"].split(" ")[0]

# Add a 'Layer Type' column
data['Layer Type'] = data.apply(determine_layer_type, axis=1)

# Add a 'Model' column
data['Model'] = data.apply(determine_model, axis=1)

# Extract batch sizes
data['Batch Size'] = data['Configuration'].str.extract(r'bs=(\d+)').astype(int)


# Get unique models and batch sizes
models = data['Model'].unique()
batch_sizes = sorted(data['Batch Size'].unique())
layer_types = data['Layer Type'].unique()

# Prepare subplots
prepare_figure(size_fraction=1.0)
fig, axes = plt.subplots(len(models), len(batch_sizes), sharey=True)
fig.suptitle("SLiM Speedup on RTX-3090")

# Colors for the bars - Dark Blue for FP16 LoRA, Red for INT4 LoRA
colors = ["#1f77b4", "#d62728"]

# Plot the data
for i, model in enumerate(models):
    for j, batch_size in enumerate(batch_sizes):
        ax = axes[i, j]
        # Filter data for the specific model and batch size
        subset = data[(data['Model'] == model) & (data['Batch Size'] == batch_size)]
        if not subset.empty:
            fpt16_speedups = []
            int4_speedups = []
            for layer_type in layer_types:
                layer_subset = subset[subset['Layer Type'] == layer_type]
                if not layer_subset.empty:
                    fpt16_speedups.append(layer_subset['lora_linear_fp16_speedup'].values[0])
                    int4_speedups.append(layer_subset['lora_linear_marlin_int4_speedup'].values[0])
                else:
                    fpt16_speedups.append(0)
                    int4_speedups.append(0)

            # Plot the bars
            bar_width = 0.4  # Width of the bars

            ax.bar(
                np.arange(len(layer_types)) - bar_width / 2,
                fpt16_speedups,
                color=colors[0],
                width=bar_width,
                label='FP16 LoRA'
            )
            ax.bar(
                np.arange(len(layer_types)) + bar_width / 2,
                int4_speedups,
                color=colors[1],
                width=bar_width,
                label='INT4 LoRA'
            )

        # Set subplot title
        if i == 0:
            ax.set_title(f"Batch Size {batch_size}", fontsize=10)

        # Set y-axis label only for the first column
        if j == 0:
            ax.set_ylabel(model)

        # Set x-axis labels
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(layer_types, rotation=7, fontsize=7)
        post_process_figure(ax)

# Add a legend
handles = [
    plt.Line2D([0], [0], color=colors[0], label='FP16 LoRA'),
    plt.Line2D([0], [0], color=colors[1], label='INT4 LoRA')
]
fig.legend(handles=handles, loc='upper left', fontsize=8)

# Adjust layout
# plt.show()
plt.savefig("results/rtx_speedup.pdf", bbox_inches='tight')