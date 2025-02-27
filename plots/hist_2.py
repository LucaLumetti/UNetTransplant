import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# Dati aggiornati
labels = ['Liver', 'Spleen', 'Kidney',  'Stomach', 'Overall']
sharp_avg_data = [0.85, 0.77, 0.55,  0.16, np.mean([0.85, 0.77, 0.55,  0.16])]
flat_avg_data = [0.91, 0.82,0.61,  0.63, np.mean([0.91, 0.82,0.61,  0.63])]
sharp_ties_data = [0.82, 0.73,  0.59,  0.14, np.mean([0.82, 0.73,  0.59,  0.14])]
flat_ties_data = [ 0.89, 0.66  ,0.34,  0.58, np.mean([ 0.89, 0.66  ,0.34,  0.58])]

# Creazione della colormap
color_map = plt.get_cmap('coolwarm')

# Creazione del grafico
fig, ax = plt.subplots(figsize=(12, 5))

# Larghezza delle barre
bar_width = 0.2
index = np.arange(len(labels))

# Creazione delle barre con zorder superiore a quello delle linee di grid
ax.bar(index - 0.60*bar_width, sharp_avg_data, bar_width, label='Sharp, Avg', color=color_map(0.65), edgecolor='black', zorder=3)
ax.bar(index - 1.60*bar_width, flat_avg_data, bar_width, label='Flat, Avg', color=color_map(0.8), edgecolor='black', hatch="//", zorder=3)
ax.bar(index + 1.60*bar_width, sharp_ties_data, bar_width, label='Sharp, Ties', color=color_map(0.3), edgecolor='black', zorder=3)
ax.bar(index + 0.60*bar_width, flat_ties_data, bar_width, label='Flat, Ties', color=color_map(0.2), edgecolor='black', hatch="//", zorder=3)

# Impostazioni degli assi
ax.set_ylabel('Dice Score', fontsize=20, weight='bold')
ax.set_xticks(index)
ax.set_xticklabels(labels, fontsize=22)
ax.set_ylim(0, 1.0)


fig.text(0.42, 0.92, 'BTCV Abdomen', va='center', rotation='horizontal', fontsize=22, weight='bold')
# Rendi "Overall" in grassetto
labels = ax.get_xticklabels()
labels[-1].set_weight('bold')  # Imposta il grassetto solo per "Overall"

# Aggiungi linee orizzontali tratteggiate dietro alle barre
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5, zorder=0)

# Rimuovere gli assi in alto e a destra
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Dimensioni dei tick dell'asse y
ax.tick_params(axis='y', labelsize=17)

# Legenda
legend_elements = [
    Patch(facecolor=color_map(0.8), label='Flat, Avg', hatch='//'),
    Patch(facecolor=color_map(0.6), label='Sharp, Avg'),
    Patch(facecolor=color_map(0.3), label='Flat, Ties', hatch='//'),
    Patch(facecolor=color_map(0.2), label='Sharp, Ties'),

]
ax.legend(handles=legend_elements, loc='upper right', fontsize=18, ncol=2, title_fontsize='15', fancybox=False, edgecolor='black')
plt.show()

# Salvataggio del grafico in formato PDF
fig.savefig('hist2.pdf',dpi=600, bbox_inches='tight')
