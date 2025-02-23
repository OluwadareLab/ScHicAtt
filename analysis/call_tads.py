import pandas as pd
import subprocess

INPUT_FILEPATH = "/home/mohit/Documents/project/ScHiCAtt/dedoc2/matrix"
SUB_FOLDERS = ["self", "diff_cell"]
CHROMOSOMES = ["chr12",  "chr12"]
ALGORITHMS = ["higashi", "schicluster"]
RESOLUTION = 40000
OUTPUT_PATH = "/home/mohit/Documents/project/ScHiCAtt/dedoc2/tads"

# Detete TADs 
for subfolder, chromosome in zip(SUB_FOLDERS, CHROMOSOMES):
    true_file = f"{INPUT_FILEPATH}/{subfolder}_{chromosome}.txt"
    output_file = f"{OUTPUT_PATH}/{subfolder}_{chromosome}"
    try:
        print(f"Running deDoc2 for {subfolder}_{chromosome}.txt")
        subprocess.call(['java', '-jar', '/home/mohit/Documents/project/ScHiCAtt/dedoc2/deDoc2.jar',
                        '-inputfile', true_file, '-binsize', '40', '-outputfile', output_file])
        for algorithm in ALGORITHMS:
            input_matrix = f"{INPUT_FILEPATH}/{subfolder}_{algorithm}_{chromosome}.txt"
            output_file = f"{OUTPUT_PATH}/{subfolder}_{algorithm}_{chromosome}"
            print(
                f"Running deDoc2 for {subfolder}_{algorithm}_{chromosome}.txt")
            subprocess.call(['java', '-jar', '/home/mohit/Documents/project/ScHiCAtt/dedoc2/deDoc2.jar',
                            '-inputfile', input_matrix, '-binsize', '40', '-outputfile', output_file])
    except Exception as e:
        print(e)
        break


# Format TADs  
INPUT_FILEPATH = "/home/mohit/Documents/project/ScHiCAtt/dedoc2/tads"
SUB_FOLDERS = ["self", "diff_cell"]
CHROMOSOMES = ["chr12",  "chr12"]
ALGORITHMS = ["higashi", "schicluster"]
RESOLUTION = 40000
OUTPUT_PATH = "/home/mohit/Documents/project/ScHiCAtt/dedoc2/tads"


def get_tads(file, chr, output_file):
    with open(output_file, "w") as fw:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                if int(line[0]) != int(line[-1]):
                    output = f"{chr}\t{int(line[0])*RESOLUTION}\t{int(line[-1])*RESOLUTION}\n"
                    fw.write(output)
        f.close()
    fw.close()


for subfolder, chromosome in zip(SUB_FOLDERS, CHROMOSOMES):
    print(f"Running deDoc2 for {subfolder}_{chromosome}.window.TAD")
    true_file = f"{INPUT_FILEPATH}/{subfolder}_{chromosome}.window.TAD"
    output_file = f"{OUTPUT_PATH}/{subfolder}_{chromosome}.tads"
    get_tads(true_file, chromosome, output_file)
    for algorithm in ALGORITHMS:
        print(
            f"Running deDoc2 for {subfolder}_{algorithm}_{chromosome}.window.TAD")
        input_matrix = f"{INPUT_FILEPATH}/{subfolder}_{algorithm}_{chromosome}.window.TAD"
        output_file = f"{OUTPUT_PATH}/{subfolder}_{algorithm}_{chromosome}.tads"
        get_tads(input_matrix, chromosome, output_file)
