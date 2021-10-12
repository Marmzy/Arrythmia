#!/usr/bin/env bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage(){
    if [ -n "$1" ]; then
        echo -e "${RED}→ $1${CLEAR}"
    fi
    echo "Usage: $0 [-h help] [-v verbose] [-d data] [-l lr] [-e epochs] [-r rescale] [-i imbalance] [-m metric]"
    echo " -h, --help       Print this help and exit"
    echo " -v, --verbose    Print verbose messages"
    echo " -d, --data       Name of the data directory"
    echo " -l, --lr         ADAM learning rate"
    echo " -e, --epochs     Number of epochs"
    echo " -r, --rescale    Rescale weights for class imbalance"
    echo " -i, --imbalance  Strategy to alleviate class imbalance (sampling | weights)"
    echo " -m, --metric     Evaluation metric (accuracy | sensitivity)"
    echo ""
    echo "Example: $0 -d data -l 0.0001 -e 25 -r -i sampling -m accuracy"
    exit 1
}

#Parsing command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -v|--verbose) VERBOSE=true ;;
        -d|--data) INPUT="$2"; shift ;;
        -l|--lr) LR="$2"; shift ;;
        -e|--epochs) EPOCHS="$2"; shift ;;
        -r|--rescale) RESCALE=true ;;
        -i|--imbalance) IMBALANCE="$2"; shift ;;
        -m|--metric) METRIC="$2"; shift ;;
    esac
    shift
done

#Verifying arguments
if [ -z "$VERBOSE" ]; then VALUE_V=false; else VALUE_V=true; fi;
if [ -z "$INPUT" ]; then usage "Input directory name is not specified"; else IN_DIR=$INPUT; fi;
if [ -z "$LR" ]; then usage "Learning rate if not specified"; else VALUE_R=$LR; fi
if [ -z "$EPOCHS" ]; then usage "Number of epochs is not specified"; else VALUE_E=$EPOCHS; fi;
if [ -z "$RESCALE" ]; then VALUE_R=false; VALUE_R=true; fi;
if [ -z "$IMBALANCE" ]; then usage "No imbalance strategy is specified"; else VALUE_I=$IMBALANCE; fi;
if [ -z "$METRIC" ]; then usage "Evaluation metric is not specified"; else VALUE_M=$METRIC; fi;


#Asserting the directory exists
if [ ! -d ${PWD%/*}/$IN_DIR ]; then
    echo "The specified directory name '${IN_DIR}' does not exist in ${PWD%/*}"
    exit 1
fi

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

#Training the model
python3 ${PWD%/*}/src/data_train.py \
        --verbose $VALUE_V \
        --data $IN_DIR \
        --normalised $VALUE_N \
        --denoised $VALUE_D \
        --lr $VALUE_R \
        --epochs $VALUE_E \
        --weight $WEIGHTS \
        --sampling $SAMPLING \
        --metric $VALUE_M
