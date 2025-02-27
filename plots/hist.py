import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# Dati aggiornati
labels = ['Mandible', 'Pharinx', 'Teeth', 'Canals', 'Overall']
sharp_avg_data = [0.854431, 0.20286855, 0.56661206, 0.01749696, np.mean([0.854431, 0.20286855, 0.56661206, 0.01749696])]
flat_avg_data = [0.89862615, 0.7690184, 0.6791316, 0.47925806, np.mean([0.89862615, 0.7690184, 0.6791316, 0.47925806])]
sharp_ties_data = [0.8235596, 0.03978382, 0.49895245, 0.04126582, np.mean([0.8235596, 0.03978382, 0.49895245, 0.04126582])]
flat_ties_data = [0.8814558, 0.7515119, 0.6480092, 0.53822756, np.mean([0.8814558, 0.7515119, 0.6480092, 0.53822756])]

# Creazione della colormap
color_map = plt.get_cmap('coolwarm')

# Creazione del grafico
fig, ax = plt.subplots(figsize=(12, 5))

# Larghezza delle barre
bar_width = 0.2
index = np.arange(len(labels))

# Creazione delle barre
ax.bar(index - 1.60*bar_width, flat_avg_data, bar_width, label='Flat, Avg', color=color_map(0.8), edgecolor='black', hatch="//", zorder=3)
ax.bar(index - 0.60*bar_width, sharp_avg_data, bar_width, label='Sharp, Avg', color=color_map(0.65), edgecolor='black', zorder=3)
ax.bar(index + 0.60*bar_width, flat_ties_data, bar_width, label='Flat, Ties', color=color_map(0.2), edgecolor='black', hatch="//", zorder=3)
ax.bar(index + 1.60*bar_width, sharp_ties_data, bar_width, label='Sharp, Ties', color=color_map(0.3), edgecolor='black', zorder=3)

# Impostazioni degli assi
ax.set_ylabel('Dice Score', fontsize=20, weight='bold')
ax.set_xticks(index)
ax.set_xticklabels(labels, fontsize=22)
ax.set_ylim(0, 1.0)

fig.text(0.42, 0.92, 'ToothFairy2', va='center', rotation='horizontal', fontsize=22, weight='bold')

# Rendi "Overall" in grassetto
labels = ax.get_xticklabels()
labels[-1].set_weight('bold')  # Imposta il grassetto solo per "Overall"

# Aggiungi linee orizzontali tratteggiate
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5, zorder=0)

# Rimozione degli assi non necessari
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Dimensioni dei tick dell'asse y
ax.tick_params(axis='y', labelsize=17)

# Legenda
legend_elements = [
    Patch(facecolor=color_map(0.8), label='Flat, Avg', hatch='//'),
    Patch(facecolor=color_map(0.65), label='Sharp, Avg'),
    Patch(facecolor=color_map(0.3), label='Sharp, Ties'),
    Patch(facecolor=color_map(0.2), label='Flat, Ties', hatch='//'),
]
#ax.legend(handles=legend_elements, loc='upper right', fontsize=12, ncol=2, title_fontsize='15', fancybox=True, edgecolor='black')

plt.show()

# Salvataggio del grafico
fig.savefig('hist.pdf', dpi=600, bbox_inches='tight')