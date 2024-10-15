# AortaSeg_2024
Contribution to the AortaSeg Challenge (MICCAI 2024) by lWM.

# How to reproduce?

## Option 1:

* Download the models (the link contains all the models used for the final submission - after the final retraining): [Link](https://drive.google.com/drive/folders/1VYhXwpf2IMCBl0Ij_9UXJzUKZOjdjhIT?usp=sharing)
* Save the models to a chosen folder and copy the absolute path
* Navigate to the [*/reproduce*](./src/reproduce) folder
* There is a file [reproduce.ipynb](./src/reproduce/reproduce.ipynb). Set the three variables:
  * 'input_path' - direct path to the input CT volume
  * 'output_path' - direct path where the segmentation mask should be saved
  * 'models_path' - path to the directory where the models are saved
* Run the inference by sequentially running all the cells in the notebook. To run the inference you need GPU with at least 10 GB of VRAM.
* If you want to run the inference for numerous files - just introduce for loop to the notebook and iteratve over numerous files.

## Option 2:

Directly download and use the Docker used for the final submission: [Link](TODO)

The Docker follows all the conventions from the [Grand Challenge](https://grand-challenge.org/) platform - use it as a Grand Challenge Algorithm.

## Option 3:
Request access to the algorithm on the [Grand Challenge](https://grand-challenge.org/algorithms/aortasegsimple/) platform and test the method directly using the GC infrastructure.

# Technical description

The technical description of the proposed method is available at: [Link](TODO)
