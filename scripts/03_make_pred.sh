#!/usr/bin/env bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage(){
    if [ -n "$1" ]; then
        echo -e "${RED}→ $1${CLEAR}"
    fi
    echo "Usage: $0 [-h help] [-v verbose] [-d data] [-i imbalance] [-k kfold]"
    echo " -h, --help       Print this help and exit"
    echo " -v, --verbose    Print verbose messages"
    echo " -d, --data       Name of the data directory"
    echo " -i, --imbalance  Strategy to alleviate class imbalance (sampling | weights)"
    echo " -k, --kfold      Number of folds the training dataset was split into"
    echo ""
    echo "Example: $0 -d data -i sampling -k 5"
    exit 1
}

#Parsing command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -v|--verbose) VERBOSE=true ;;
        -d|--data) INPUT="$2"; shift ;;
        -i|--imbalance) IMBALANCE="$2"; shift ;;
        -k|--kfold) KFOLD="$2"; shift ;;
    esac
    shift
done

#Verifying arguments
if [ -z "$VERBOSE" ]; then VALUE_V=false; else VALUE_V=true; fi;
if [ -z "$INPUT" ]; then usage "Input directory name is not specified"; else IN_DIR=$INPUT; fi;
if [ -z "$IMBALANCE" ]; then usage "No imbalance strategy is specified"; else VALUE_I=$IMBALANCE; fi;
if [ -z "$KFOLD" ]; then usage "Number of kfolds is not specified"; else VALUE_K=$KFOLD; fi;


#Asserting the directory and exists
if [ ! -d ${PWD%/*}/$IN_DIR ]; then
    echo "The specified directory name '${IN_DIR}' does not exist in ${PWD%/*}"
    exit 1
fi

#Getting the trained model stem name
STEM=$(ls -lh ${PWD%/*}/$IN_DIR/output/* | grep "pkl$" | head -n 1 | awk '{print $9}' | cut -d'/' -f 8 | sed -e "s/_fold.*//")

#Checking the available datasets
NORM=$(ls -lh ${PWD%/*}/$IN_DIR/train/* | grep "normalised")
NOISE=$(ls -lh ${PWD%/*}/$IN_DIR/train/* | grep "denoised")

if [ -z "$NORM" ]; then VALUE_N=false; else VALUE_N=true; fi;
if [ -z "$NOISE" ]; then VALUE_D=false; else VALUE_D=true; fi;

#Passing the sampling strategy correctly
if [ $VALUE_I == "sampling" ]; then
    SAMPLING=true
    WEIGHTS=false
else
    SAMPLING=false
    WEIGHTS=true
fi

#Evaluating the trained models
python3  ${PWD%/*}/src/data_eval.py \
        --verbose $VALUE_V \
        --indir $IN_DIR \
        --infiles $STEM \
        --normalised $VALUE_N \
        --denoised $VALUE_D \
        --kfold $VALUE_K \
        --weight $WEIGHTS \
        --sampling $SAMPLING
