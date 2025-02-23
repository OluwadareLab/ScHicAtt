from heatmap import *
from moc import *
import pandas as pd

INPUT_FILEPATH = "/home/mohit/Documents/project/ScHiCAtt/dedoc2/tads/"
MATRIX_FILEPATH = "/home/mohit/Documents/project/ScHiCAtt/dedoc2/matrix/"
# FILENAMES = [
#     ["self_chr12",
#      "self_schicatt_chr12",
#      "self_schicedrn_chr12",
#      "self_deephic_chr12",
#      "self_loopenhance_chr12"],
#     ["diff_cell_chr12",
#      "diff_cell_schicatt_chr12",
#      "diff_cell_schicedrn_chr12",
#      "diff_cell_deephic_chr12",
#      "diff_cell_loopenhance_chr12"]
#     # ["diff_species_chrX",
#     #  "diff_species_deephic_chrX",
#     #  "diff_species_loopenhance_chrX",
#     #  "diff_species_schicatt_chrX",
#     #  "diff_species_schicedrn_chrX"],
# ]
FILENAMES = [
    ["self_chr12",
     "self_higashi_chr12",
     "self_schicluster_chr12"],
    ["diff_cell_chr12",
     "diff_cell_higashi_chr12",
     "diff_cell_schicluster_chr12"]
    # ["diff_species_chrX",
    #  "diff_species_deephic_chrX",
    #  "diff_species_loopenhance_chrX",
    #  "diff_species_schicatt_chrX",
    #  "diff_species_schicedrn_chrX"],
]
CHROMOSOMES = [
    "chr12",  "chr12"]
ALGORITHMS = ["Higashi", "scHiCluster"]

OUTPUT_PATH = "/home/mohit/Documents/project/ScHiCAtt/dedoc2/plots/"

for fnames, chromosome in zip(FILENAMES, CHROMOSOMES):
    mat = MATRIX_FILEPATH+fnames[0]+".txt"
    # true_file = INPUT_FILEPATH + fnames[0] + ".bed"
    # true_tads_m = read_(true_file)
    matf= np.loadtxt(mat)
    for idx in range(1, 3, 1):
        ref_file = INPUT_FILEPATH + fnames[idx] + ".tads"
        print(f'Processing -> {ref_file}')
        # ref_tads_m = read_(ref_file)
        # moc_score = get_MoC(tads=ref_tads_m, true_tads=true_tads_m)

        out_file = OUTPUT_PATH + fnames[idx]
        vis = Triangle(mat, 40000, chromosome.replace(
            'chr', ''), 20000000, 24000000, title=ALGORITHMS[idx-1])
        vis.matrix_plot()
        vis.plot_TAD(ref_file, linewidth=2)
        print(f'Writing -> {out_file}')
        vis.outfig(out_file)
