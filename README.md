# Arrythmia

A project in which I train model with the classic ResNet34 architecture to classify Normal Ectopic Heartbeats (N) from Ventricular Ectopic Heartbeats (V).
The code allows for the handling of data imbalance using weights or undersampling, can preprocess the data automatically and allows the setting of various
hyperparameters.

This project was made on Ubuntu 18.04 using Windows Subsystem for Linux 2.

## Starting off

To run the code, a suitable development environment must be set up. The GNU/Linux environment needs to have Python3 (>3.8.1), along with the following repositories:
[requirements.txt](https://github.com/Marmzy/Arrythmia/blob/main/requirements.txt). Ideally a virtual environment is setup using Anaconda or pyenv to ensure the requirements for this project don't interfere with any other.

All scripts necessary for the analysis can be found in the 'scripts' directory.

## Preprocessing

To download and preprocess the dataset, run: [`01_dataprep.sh`](https://github.com/Marmzy/Arrythmia/blob/main/scripts/01_dataprep.sh)

```bash
Usage: 01_dataprep.sh [-h help] [-v verbose] [-o output] [-n norm] [-d denoise] [-i imbalance] [-k kfold]
 -h, --help       Print this help and exit
 -v, --verbose    Print verbose messages
 -o, --output     Name of output directory where data will be stored
 -n, --norm       Normalise the dataset
 -d, --denoise    Denoise the dataset
 -i, --imbalance  Strategy to alleviate class imbalance (sampling | weights)
 -k, --kfold      Number of folds to split the training dataset into

Example: 01_dataprep.sh -o data -n -d -i sampling -k 5
```

01_dataprep.sh will automatically download the [MIT-BIH Arrythmia dataset](https://www.physionet.org/content/mitdb/1.0.0/)<sup>1,2</sup> if it hasn't been downloaded before
and will prepare the output directory structure. In this step one can transform the data through normalisation and/or wavelet denoising. Furthermore, a strategy
to alleviate class imbalance must be provided:
 - sampling (random undersampling of the majority class)
 - weights (adding weights to the loss function)

The output of the script will be the creation of train, val and test datasets in the output directory.

## Training

To train the model, run [`02_model_train_eval.sh`](https://github.com/Marmzy/Arrythmia/blob/main/scripts/02_model_train_eval.sh)

```bash
Usage: 02_model_train_eval.sh [-h help] [-v verbose] [-d data] [-l lr] [-e epochs] [-b batch] [-p processing] [-i imbalance] [-m metric] [-k kfold]
 -h, --help       Print this help and exit
 -v, --verbose    Print verbose messages
 -d, --data       Name of the data directory
 -l, --lr         ADAM learning rate
 -e, --epochs     Number of epochs
 -b, --batch      Minibatch size
 -p, --processing Processing of the data (none | norm | denoise | both)
 -i, --imbalance  Strategy to alleviate class imbalance (sampling | weights)
 -m, --metric     Evaluation metric (accuracy | sensitivity)
 -k, --kfold      Number of folds the training dataset was split into

Example: 02_model_train_eval.sh -d data -l 0.0001 -e 10 -b 64 -p both -i sampling -m accuracy -k 5
```

02_model_train_eval.sh will train the model for each k-fold with the specified hyperparameters on the training dataset and immediately evaluate it on the
validation dataset. It reports back the training and validation loss and will save the model with the highest score for the specified metric (accuracy or
sensitivity).
One can optionally run the [`DataVisualisation.ipynb`](https://github.com/Marmzy/Arrythmia/blob/main/src/DataVisualisation.ipynb) notebook to
visualise the training process.

The output of the script is a subdirectory in the output directory, containing logs of the training process and the best models for each fold.

## Evaluation

To evaluate the model, run [`03_make_pred.sh`](https://github.com/Marmzy/Arrythmia/blob/main/scripts/03_make_pred.sh)

```bash
Usage: 03_make_pred.sh [-h help] [-v verbose] [-d data] [- n name]
 -h, --help       Print this help and exit
 -v, --verbose    Print verbose messages
 -d, --data       Name of the data directory
 -n, --name       Name of the directory containing the models to evaluate

Example: 03_make_pred.sh -d data -n ResNet34_sampled_lr0.001_decay0.0_epochs10_batch64_accuracy
```

03_make_pred.sh will make and save predictions with the k trained models and score them across severall metrics.

---

<sup>1</sup>: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)

<sup>2</sup>:  	Goldberger, A., et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220." (2000).
