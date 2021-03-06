#!/usr/bin/env bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage(){
    if [ -n "$1" ]; then
        echo -e "${RED}→ $1${CLEAR}"
    fi
    echo "Usage: $0 [-h help] [-v verbose] [-d data] [- n name]"
    echo " -h, --help       Print this help and exit"
    echo " -v, --verbose    Print verbose messages"
    echo " -d, --data       Name of the data directory"
    echo " -n, --name       Name of the directory containing the models to evaluate"
    echo ""
    echo "Example: $0 -d data -n ResNet34_sampled_lr0.001_decay0.0_epochs10_batch64_accuracy"
    exit 1
}

#Parsing command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -v|--verbose) VERBOSE=true ;;
        -d|--data) IN_DIR="$2"; shift ;;
        -n|--name) NAME="$2"; shift ;;
    esac
    shift
done

#Verifying arguments
if [ -z "$VERBOSE" ]; then VALUE_V=false; else VALUE_V=true; fi;
if [ -z "$IN_DIR" ]; then usage "Data directory name is not specified"; fi;
if [ -z "$NAME" ]; then usage "Analysis directory name is not specified"; fi;


#Asserting the directory and exists
if [ ! -d ${PWD%/*}/$IN_DIR ]; then
    echo "The specified directory name '${IN_DIR}' does not exist in ${PWD%/*}"
    exit 1
fi

#Correctly passing on sampling parameters
if [[ $NAME =~ "sampled" ]]; then
    VALUE_S=true
    VALUE_W=false
else
    VALUE_S=false
    VALUE_W=true
fi

#Correctly passing on data transformation parameters
if [[ $NAME =~ "normalised" ]]; then VALUE_N=true; else VALUE_N=false; fi;
if [[ $NAME =~ "denoised" ]]; then VALUE_D=true; else VALUE_D=false; fi;

#Correctly passing on the number folds
VALUE_K=$(ls -lh ${PWD%/*}/$IN_DIR/output/$NAME | awk '{print $9}' | grep ".log" | wc -l)

#Evaluating the trained models
python3  ${PWD%/*}/src/data_eval.py \
        --verbose $VALUE_V \
        --indir $IN_DIR \
        --infiles $NAME \
        --normalised $VALUE_N \
        --denoised $VALUE_D \
        --kfold $VALUE_K \
        --weight $VALUE_W \
        --sampling $VALUE_S
