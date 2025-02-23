import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def add_annotations(ax, bars, fontsize=26):
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        fontweight = 'bold' if i == 0 else 'normal'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f'{yval:.2f}',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            fontweight=fontweight  # Set the font weight
        )


def insulation_score(m, windowsize=5000000, res=40000):
    windowsize_bin = int(windowsize / res)
    score = np.ones((m.shape[0]))
    for i in range(0, m.shape[0]):
        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.sum(m[max(0, i - windowsize_bin): i, i + 1: min(m.shape[0] - 1, i + windowsize_bin + 1)]) / (np.sum(
                m[max(0, i - windowsize_bin):min(m.shape[0], i + windowsize_bin + 1),
                  max(0, i - windowsize_bin):min(m.shape[0], i + windowsize_bin + 1)]))
            if np.isnan(v):
                v = 1.0

        score[i] = v
    return score


MATRIX_FILEPATH = "/home/mohit/Documents/project/ScHiCAtt/dedoc2/matrix/"
FILENAMES = [
    ["self_chr12",
     "self_schicatt_chr12",
     "self_schicedrn_chr12",
     "self_deephic_chr12",
     "self_loopenhance_chr12",
     "self_higashi_chr12",
     "self_schicluster_chr12"],
    ["diff_cell_chr12",
     "diff_cell_schicatt_chr12",
     "diff_cell_schicedrn_chr12",
     "diff_cell_deephic_chr12",
     "diff_cell_loopenhance_chr12",
     "diff_cell_higashi_chr12",
     "diff_cell_schicluster_chr12"],
    # ["diff_species_chrX",
    #  "diff_species_deephic_chrX",
    #  "diff_species_loopenhance_chrX",
    #  "diff_species_schicatt_chrX",
    #  "diff_species_schicedrn_chrX"],
]
CHROMOSOMES = ["chr12",  "chr12"]
ALGORITHMS = ["ScHiCAtt", "ScHiCEDRN", "DeepHiC", "Loopenhance", "Higashi", "scHiCluster"]

l2norms = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
i = 0
for fnames in FILENAMES:
    original_mat = MATRIX_FILEPATH+fnames[0]+".txt"
    original_mat = np.loadtxt(original_mat)
    insulation_a = insulation_score(original_mat)
    for idx in range(1, 7, 1):
        mat = MATRIX_FILEPATH+fnames[idx]+".txt"
        mat = np.loadtxt(mat)
        insulation_b = insulation_score(mat)
        l2norms[i][idx-1] = np.linalg.norm(insulation_a-insulation_b)
        print(f"L2 norm of {fnames[idx]}: {l2norms[i][idx-1]}")
    i = i+1

colors = sns.color_palette("bright", 4)
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_data, title in zip(axes.flat, l2norms, ["Same Cell Same Species", "Different Cell Same Species"]):
    bars = sns.barplot(x=ALGORITHMS, y=y_data, palette=colors, ax=ax)
    add_annotations(ax, bars.patches, 10)
    # ax.get_xaxis().set_visible(False)
    ax.set_title(title)
    ax.set_ylabel("L2 norm of insulation score")

plt.tight_layout(rect=[0, 0, 1, 1.05])
plt.savefig(f"l2norm.png",
            dpi=600, bbox_inches="tight")
