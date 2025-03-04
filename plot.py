# for each folder in output_UNetMergingmerge create a plot using the name of the folder as the title and using data from the Task1+Task2.npy file
import os

import matplotlib.pyplot as plt
import numpy as np

for folder in os.listdir("./output_UNetMergingmerge"):
    tasks = folder.split("_")[0].split("+")
    task_0 = tasks[0]
    task_1 = tasks[1]
    data = np.load(f"./output_UNetMergingmerge/{folder}/Task1+Task2.npy")

    # Assuming the structure of 'data' and that we need only the second half for the positive quadrant
    half_data = data[
        21:, 21:
    ]  # Adjust indexing as necessary based on your specific data layout

    # Set up the colormap
    cmap = plt.get_cmap("coolwarm")
    # make color map more differentiable
    cmap = plt.get_cmap("coolwarm", 16)

    plt.figure(figsize=(10, 8))

    # Plot the heatmap for the positive quadrant
    # tupes of interpolations: ['kaiser', 'sinc', 'quadric', 'none', 'bilinear', 'nearest', 'bicubic', 'spline36', 'bessel', 'lanczos', 'spline16', 'blackman', 'gaussian', 'hermite', 'hanning', 'catrom', 'mitchell', 'hamming', 'auto', 'antialiased']
    data_im_reduced_contours_star = plt.imshow(
        half_data,
        extent=[0, 2, 0, 2],
        origin="lower",
        cmap=cmap,
        interpolation="bicubic",
        vmax=1.0,
    )
    # data_im_reduced_contours_star = plt.imshow(half_data, extent=[0, 2, 0, 2], origin='lower', cmap=cmap, vmax=1.0)

    # Overlay fewer contour lines focusing on levels from 0.40 to 0.80 with increased gaps
    contour_spaced_star = plt.contour(
        np.linspace(0, 2, half_data.shape[1]),
        np.linspace(0, 2, half_data.shape[0]),
        half_data,
        levels=np.linspace(0.4, 0.9, 6),
        colors="black",
        linewidths=0.7,
        alpha=0.7,
    )

    # Add labels to contour lines
    plt.clabel(contour_spaced_star, inline=True, fontsize=10, fmt="%.2f")

    # Calculate the position of the maximum value within the focused range
    max_value = np.max(half_data)
    max_indices = np.unravel_index(np.argmax(half_data), half_data.shape)
    max_x, max_y = (
        max_indices[1] / 10,
        max_indices[0] / 10,
    )  # convert indices to scale based on your plot's extent

    # Mark the maximum value with a red star
    plt.plot(
        max_x,
        max_y,
        "*",
        markersize=12,
        markeredgewidth=1,
        color="red",
        markeredgecolor="black",
    )

    # plot marker at x=1 and y=1
    plt.plot(
        1,
        1,
        "+",
        markersize=12,
        markeredgewidth=1,
        color="green",
        markeredgecolor="black",
    )
    # plot also the value at x=1 and y=1
    # plt.text(1, 1+0.015, f'{half_data[10, 10]:.2f}', fontsize=10, ha= 'center', va='bottom', color='black')

    # plot also the maximum value above the star with a small pad
    plt.text(
        max_x,
        max_y + 0.015,
        f"{max_value:.2f}",
        fontsize=10,
        ha="center",
        va="bottom",
        color="black",
    )

    # Create colorbar for the full range of the data
    cbar = plt.colorbar(data_im_reduced_contours_star, label="Avg. Dice", format="%.2f")

    # Use the alpha symbol in axis labels
    # plt.xlabel(r'$\alpha_{\mathrm{Mandible}}$')
    # plt.ylabel(r'$\alpha_{\mathrm{Canal}}$')
    # insert a space between the alpha and the task
    # plt.xlabel(r'$\alpha{\mathrm{%s}}$' % task_0)
    # plt.ylabel(r'$\alpha{\mathrm{%s}}$' % task_1)
    plt.xlabel(r"$\alpha\ \mathrm{%s}$" % task_0, fontsize=12)
    plt.ylabel(r"$\alpha\ \mathrm{%s}$" % task_1, fontsize=12)
    # plt.title('Mandible-Canals Positive Quadrant: Reduced Contour Lines with Maximum Marked')

    # Restore original axis ticks with labels
    plt.xticks(ticks=np.arange(0, 2.5, 0.5))
    plt.yticks(ticks=np.arange(0, 2.5, 0.5))

    plt.grid(False)
    plt.show()
    # save pdf
    plt.savefig(f"plots/{task_0}+{task_1}.pdf")
