import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

tasks = [
    "Mandible+Canals_Naive",
    "Pharynx+Teeth_Naive",
    "Kidney+Liver_Sharp",
    "Spleen+Stomach_Sharp",
    "Mandible+Canals_Stable1",
    "Pharynx+Teeth_Stable1",
    "Kidney+Liver_Stable",
    "Spleen+Stomach_Stable",
]

# Set up the colormap
cmap = plt.get_cmap("coolwarm")
# make color map more differentiable
cmap = plt.get_cmap("coolwarm", 16)


# Funzione per arrotondare i tick labels
def format_ticks(value, tick_number):
    return f"{value:.1f}"


# Crea il formatter
formatter = FuncFormatter(format_ticks)

# Create a figure with a specific size
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(11, 5))

# Flatten the axis array for easy iteration
axs = axs.flatten()
fontsize = 12
# reduce space beteween subplots
plt.subplots_adjust(wspace=0.15, hspace=0.1)
i = 0.0

for ax, title in zip(axs, tasks):
    # Load the data from the numpy array file
    if "Mandible+Canals" in title or "Phar" in title:
        if i < 4:
            data_path = f"PrevisioniMeteoSharpToothFairy/"
            data = np.load(f"{data_path}{title}_TaskVector/Task1+Task2.npy")
        elif i < 8:
            data_path = f"PrevisioniMeteoFlatAvgToothFairy/"
            data = np.load(f"{data_path}{title}_TaskVector/Task1+Task2.npy")
        else:
            data_path = f"output_UNetMergingmerge/"
            data = np.load(f"{data_path}{title}_TaskVectorTies/Task1+Task2.npy")

    else:
        if i < 8:
            data_path = f"PrevisioniMeteoAbdomen/Abdomen/"
            data = np.load(f"{data_path}{title}_TaskVector/Task1+Task2.npy")
        else:
            data_path = f"PrevisioniMeteoAbdomen/Abdomen/"
            data = np.load(f"{data_path}{title}_TaskVectorTies/Task1+Task2.npy")

    # Assume the structure of 'data' and that we need only the second half for the positive quadrant
    if "output_UNetMergingmerge/" in data_path and "Stable" in title:
        half_data = data[21:, 21:]
    elif "output_UNetMergingmerge/" in data_path and "Naive" in title:
        half_data = data[1:, 1:]
        half_data[19, 19] = 0.0
    elif "PrevisioniMeteoAbdomen/" in data_path and "Stable" in title and i >= 8:
        half_data = data[1:, 1:]
        half_data[19, 19] = 0.0
    elif "PrevisioniMeteoAbdomen/" in data_path and "Stable" in title and i < 8:
        half_data = data[:11, :11]
        half_data[10, 10] = 0.0
    else:
        half_data = data

    # Plot the heatmap for the positive quadrant

    if i < 8:
        data_im = ax.imshow(
            half_data,
            extent=[0, 1, 0, 1],
            origin="lower",
            cmap=cmap,
            interpolation="bicubic",
            vmax=1.0,
        )
        # Overlay fewer contour lines focusing on levels from 0.40 to 0.80 with increased gaps
        contour = ax.contour(
            np.linspace(0, 1, half_data.shape[0]),
            np.linspace(0, 1, half_data.shape[1]),
            half_data,
            levels=np.linspace(0.4, 0.8, 5),
            colors="black",
            linewidths=0.7,
            alpha=0.7,
        )
    else:
        data_im = ax.imshow(
            half_data,
            extent=[0, 2, 0, 2],
            origin="lower",
            cmap=cmap,
            interpolation="bicubic",
            vmax=1.0,
        )  # Overlay fewer contour lines focusing on levels from 0.40 to 0.80 with increased gaps
        contour = ax.contour(
            np.linspace(0, 2, half_data.shape[0]),
            np.linspace(0, 2, half_data.shape[1]),
            half_data,
            levels=np.linspace(0.4, 0.8, 5),
            colors="black",
            linewidths=0.7,
            alpha=0.7,
        )

    # Set the formatter for x and y axes
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    # Add labels to contour lines
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    # Calculate the position of the maximum value within the focused range
    max_value = np.max(half_data)
    max_indices = np.unravel_index(np.argmax(half_data), half_data.shape)

    if i < 4:
        max_x, max_y = (
            max_indices[1] / 20,
            max_indices[0] / 20,
        )  # convert indices to scale based on your plot's extent
    elif i < 8 and "PrevisioniMeteoFlatAvgToothFairy/" in data_path:
        max_x, max_y = (
            max_indices[1] / 12,
            max_indices[0] / 12,
        )  # convert indices to scale based on your plot's extent
    elif i < 8:
        max_x, max_y = (
            max_indices[1] / 12,
            max_indices[0] / 12,
        )  # convert indices to scale based on your plot's extent
    else:
        max_x, max_y = max_indices[1] / 10, max_indices[0] / 10

    # all markerr types: [ '+' | ',' | '.' | '1' | '2' | '3' | '4' | '<' | '>' | '8' | 's' | 'p' | '*' | 'h' | 'H' | 'x' | 'D' | 'd' | '|' | '_' ]
    # Mark the maximum value with a red star
    ax.plot(
        max_x,
        max_y,
        "*",
        markersize=12,
        markeredgewidth=1,
        color="red",
        markeredgecolor="black",
        label="Best Value",
    )

    # plot also the value of the maximum
    # if i<8:
    #    #white background
    #    ax.text(max_x, max_y-0.150, f'{max_value:.2f}', fontsize=9, ha= 'center', va='bottom', color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    # else:
    #    ax.text(max_x, max_y-0.250, f'{max_value:.2f}', fontsize=10, ha= 'center', va='bottom', color='black')

    # Mark a point of interest with a black plus
    if i < 8:
        ax.plot(
            0.5,
            0.5,
            "D",
            markersize=6,
            markeredgewidth=1,
            label="Default",
            markeredgecolor="black",
            color="white",
        )
    else:
        ax.plot(
            1,
            1,
            "D",
            markersize=5,
            markeredgewidth=1,
            label="Default",
            markeredgecolor="black",
            color="cyan",
        )

    # Set subplot title
    if i < 4:
        ax.set_title(f'{title.split("_")[0]}', fontsize=fontsize)

    # Manage tick labels
    if i % 4 != 0:
        ax.set_yticklabels([])  # Hide y tick labels for all but the first column
    if i < 4:
        ax.set_xticklabels([])  # Hide x tick labels for all but the last row
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

    i += 1

# mid_space_y = (axs[7].get_position().y0 + axs[8].get_position().y1) / 2
# fig.lines.extend([plt.Line2D([0, 1], [mid_space_y, mid_space_y], transform=fig.transFigure, color="black", linestyle="--")])

# plot a vertical line to separate the two dataset (no text)
# fig.lines.extend([plt.Line2D([0.487, 0.487], [0.12, 0.86], transform=fig.transFigure, color="grey", linestyle="--")])

fig.text(
    0.080,
    0.69,
    "Plastic - Avg",
    va="center",
    rotation="vertical",
    fontsize=12,
    weight="bold",
)
fig.text(
    0.076,
    0.30,
    "Stable - Avg",
    va="center",
    rotation="vertical",
    fontsize=12,
    weight="bold",
)

fig.text(
    0.25,
    0.953,
    "ToothFairy2",
    va="center",
    rotation="horizontal",
    fontsize=12,
    weight="bold",
)
fig.lines.extend(
    [
        plt.Line2D(
            [0.125, 0.472],
            [0.93, 0.93],
            transform=fig.transFigure,
            color="black",
            linestyle="-",
        )
    ]
)

fig.text(
    0.62,
    0.953,
    "BTCV Abdomen",
    va="center",
    rotation="horizontal",
    fontsize=12,
    weight="bold",
)
fig.lines.extend(
    [
        plt.Line2D(
            [0.503, 0.85],
            [0.93, 0.93],
            transform=fig.transFigure,
            color="black",
            linestyle="-",
        )
    ]
)
# fig.text(0.076, 0.22, 'Flat - Ties', va='center', rotation='vertical', fontsize=12, weight='bold')

# Adjust subplots layout
fig.subplots_adjust(right=0.85)
# Add vertical colorbar on the right
cbar_ax = fig.add_axes(
    [0.87, 0.125, 0.015, 0.74]
)  # Adjust the colorbar position vertically.

cbar = fig.colorbar(data_im, cax=cbar_ax, orientation="vertical")
cbar.set_label("Avg. Dice (↑)", size=14)

# Add a legend to the last plot (or any one of them)
axs[-5].legend(
    loc="upper right", fancybox=False, edgecolor="black"
)  # , frameon=True, facecolor='white', framealpha=1)

# Save the figure
plt.savefig(f"grid_plots/output.pdf", dpi=600, bbox_inches="tight")
plt.show()
