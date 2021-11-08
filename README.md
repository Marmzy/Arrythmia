# Arrythmia

A project in which I train model with the classic ResNet34 architecture to classify Normal Ectopic Heartbeats (N) from Ventricular Ectopic Heartbeats (V).
The code allows for the handling of data imbalance using weights or undersampling, can preprocess the data automatically and allows the setting of various
hyperparameters.

This project was made on Ubuntu 18.04 using Windows Subsystem for Linux 2.

## Starting off

To run the code, a suitable development environment must be set up. The GNU/Linux environment needs to have Python3 (>3.8.1), along with the following repositories:
[requirements.txt](https://github.com/Marmzy/Arrythmia/blob/main/requirements.txt). Ideally a virtual environment is setup using Anaconda or pyenv to ensure the\
requirements for this project don't interfere wiht any other.

## Preprocessing

The preprocessing script [01_dataprep.sh](https://github.com/Marmzy/Arrythmia/blob/main/scripts/01_dataprep.sh) will automatically download the [MIT-BIH Arrythmia
dataset](https://www.physionet.org/content/mitdb/1.0.0/) and prepare the output directory structure. This step will automatically 
