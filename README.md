
# ScHiCAtt: Enhancing Single-Cell Hi-C Data Resolution Using Attention-Based Models

___________________  
#### OluwadareLab, University of Colorado, Colorado Springs  
___________________

#### Developers:  
Rohit Menon  
Department of Computer Science  
University of Colorado, Colorado Springs
Email: rmenonch@uccs.edu

#### Contact:  
Oluwatosin Oluwadare, PhD
Department of Computer Science 
University of Colorado, Colorado Springs
Email: ooluwada@uccs.edu
___________________  

## Overview:
**ScHiCAtt** is a deep learning model designed to enhance the resolution of Single-Cell Hi-C contact matrices using various attention mechanisms, such as self-attention, local attention, global attention, and dynamic attention. The model leverages GAN-based training to optimize the quality of Hi-C contact maps through a composite loss function consisting of MSE, perceptual, total variation, and adversarial losses.

___________________  

## Build Instructions:

ScHiCAtt runs in a Docker-containerized environment. Follow these steps to set up ScHiCAtt.

1. Clone this repository:

```bash
git clone https://github.com/OluwadareLab/ScHiCAtt.git && cd ScHiCAtt
```

2. Pull the ScHiCAtt Docker image:

```bash
docker pull oluwadarelab/schicatt:latest
```

3. Run the container and mount the present working directory to the container:

```bash
docker run --rm --gpus all -it --name schicatt -v ${PWD}:${PWD} oluwadarelab/schicatt
```

4. You can now navigate within the container and run the model.

___________________  

## Dependencies:

All necessary dependencies are bundled within the Docker environment. The core dependencies include:

- Python 3.8
- PyTorch 1.10.0 (CUDA 11.3)
- NumPy 1.21.1
- SciPy 1.7.0
- Pandas 1.3.1
- Scikit-learn 0.24.2
- Matplotlib 3.4.2
- tqdm 4.61.2

**_Note:_** GPU usage for training and testing is highly recommended.


___________________  

## Training:

The ScHiCAtt model supports multiple attention mechanisms, such as self, local, global, and dynamic attention. To train the model, use the following command:

```bash
$ python training/train.py --epoch 300 --batch_size 8 --attention self
```

### Training Options:
- `--epoch`: Number of epochs (default: 300).
- `--batch_size`: Batch size for training (default: 8).
- `--attention`: Attention mechanism (choices: "self", "local", "global", "dynamic").

This will output `.pytorch` checkpoint files containing the trained weights. The best model is saved as `bestg.pytorch`, and the final model is saved as `finalg.pytorch`.

___________________  

## Analysis

All the analysis scripts are available at **analysis** folder
### TAD Plots
#### TAD detection with deDoc2
* We used `https://github.com/zengguangjie/deDoc2` for TAD detection from scHiC data. We used lower TAD-like domains.
* Download the **doDoc2**
* Edit the necessary variables in `call_tads.py` script such as *INPUT_FILEPATH* and adjust other things as necessary.
* Run `python3 call_tads.py`

#### TAD plot
* We used `https://xiaotaowang.github.io/TADLib/index.html` to produce the TAD figures.
* Here, we provided a sample python script `draw_tad_plots.py` to produce the plots. Update *INPUT_FILEPATH*, *MATRIX_FILEPATH*, *FILENAMES*, *CHROMOSOMES*, *ALGORITHMS*, *OUTPUT_PATH*.
* It takes matrix and TADs as input. TAD file structure should be (without heading):

| Chromosome | Start Position | End Position |
|------------|----------------|--------------|
| chr12      | 80000         | 1960000    |
| chr12      | 2000000      | 2720000    |
| chr12      | 2760000      | 4040000    |
| chr12      | 4080000      | 5480000    |
| chr12      | 5520000      | 5720000    |

* Run `python3 draw_tad_plots.py`
   
### L2 norm
* Edit draw_l2_norm.py with your filenames and paths including *MATRIX_FILEPATH*, *FILENAMES*, *CHROMOSOMES*, *ALGORITHMS*.
* It takes matrix as input.
* Run `python3 draw_l2_norm.py`

___________________  

## Cite:

If you use ScHiCAtt in your research, please cite the following:

Rohit Menon, Oluwatosin Oluwadare, *ScHiCAtt: Enhancing Single-Cell Hi-C Data with Attention Mechanisms*, [Journal Name], [Year], [DOI link if available].

___________________  

## License:

This project is licensed under the MIT License. See the `LICENSE` file for details.
