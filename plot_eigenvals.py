import matplotlib.pyplot as plt
import numpy as np

# Data
x_values = np.arange(10)
eigenvalues = {
    "TF2 Plastic": [
        6.005412,
        5.6588054,
        2.652642,
        2.5587585,
        1.8484913,
        1.7466824,
        1.6635638,
        1.5904709,
        0.7676617,
        0.7145429,
    ],
    "TF2 Stable": [
        0.02614812,
        0.01360698,
        0.01356122,
        0.01353174,
        0.00938315,
        0.00846025,
        0.00703993,
        0.00600545,
        0.00578261,
        0.00440862,
    ],
    "BTCV Plastic": [
        0.05312938,
        0.04346767,
        0.04249799,
        0.02421157,
        0.02326399,
        0.01994262,
        0.01683287,
        0.01636762,
        0.01586541,
        0.01514494,
    ],
    "BTCV Stable": [
        0.03045017,
        0.02594222,
        0.0200868,
        0.02001312,
        0.01619752,
        0.01601049,
        0.01349816,
        0.01262915,
        0.01052538,
        0.01032843,
    ],
}

# Custom Colors
# custom_colors = ["#6741d9", "#1971c2", "#e03131", "#2f9e44"]
# custom_colors = ["#0d3b66", "#13401b", "#1971c2", "#2f9e44"]
custom_colors = ["#e8590c", "#2f9e44", "#e8590c", "#2f9e44"]
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

colors = custom_colors
# colors = default_colors

# Plot
plt.figure(figsize=(2.5, 2.5))
markers = ["X", "X", "o", "o"]
linestyles = [":", "-", ":", "-"]
for i, (label, values) in enumerate(eigenvalues.items()):
    plt.plot(
        x_values,
        values,
        marker=markers[i],
        linestyle=linestyles[i],
        color=colors[i],
        label=label,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=7,
    )

# Labels and Ticks
# plt.xlabel("Eigenvalue Index", weight='bold')
# plt.ylabel("Eigenvalue", weight='bold')
plt.xticks(x_values, [f"$\\lambda_{{{i+1}}}$" for i in x_values])
plt.yscale("log")  # Set y-axis to log scale
plt.ylim(
    [
        min(min(v) for v in eigenvalues.values()) * 0.9,
        max(max(v) for v in eigenvalues.values()) * 1.2,
    ]
)

plt.legend(
    fancybox=False,
    edgecolor="black",
    fontsize="small",
    columnspacing=0.5,
    handletextpad=0.5,
    labelspacing=0.2,
)
plt.grid(True, linestyle="--", alpha=0.7)
plt.grid(axis="y", linestyle="-", alpha=0.7)
plt.grid(axis="x", linestyle="--", alpha=0.7)

plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)

plt.savefig("eigenvalues_plot.pdf", bbox_inches="tight", dpi=600)
plt.close()
