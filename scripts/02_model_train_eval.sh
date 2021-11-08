#!/usr/bin/env bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage(){
    if [ -n "$1" ]; then
        echo -e "${RED}→ $1${CLEAR}"
    fi
    echo "Usage: $0 [-h help] [-v verbose] [-d data] [-l lr] [-e epochs] [-b batch] [-p processing] [-i imbalance] [-m metric] [-k kfold]"
    echo " -h, --help       Print this help and exit"
    echo " -v, --verbose    Print verbose messages"
    echo " -d, --data       Name of the data directory"
    echo " -l, --lr         ADAM learning rate"
    echo " -e, --epochs     Number of epochs"
    echo " -b, --batch      Minibatch size"
    echo " -p, --processing Processing of the data (none | norm | denoise | both)"
    echo " -i, --imbalance  Strategy to alleviate class imbalance (sampling | weights)"
    echo " -m, --metric     Evaluation metric (accuracy | sensitivity)"
    echo " -k, --kfold      Number of folds the training dataset was split into"
    echo ""
    echo "Example: $0 -d data -l 0.0001 -e 10 -b 64 -p both -i sampling -m accuracy -k 5"
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
        -b|--batch) BATCH="$2"; shift ;;
        -p|--processing) PROCESSING="$2"; shift ;;
        -i|--imbalance) IMBALANCE="$2"; shift ;;
        -m|--metric) METRIC="$2"; shift ;;
        -k|--kfold) KFOLD="$2"; shift ;;
    esac
    shift
done

#Verifying arguments
if [ -z "$VERBOSE" ]; then VALUE_V=false; else VALUE_V=true; fi;
if [ -z "$INPUT" ]; then usage "Input directory name is not specified"; else IN_DIR=$INPUT; fi;
if [ -z "$LR" ]; then usage "Learning rate if not specified"; else VALUE_R=$LR; fi
if [ -z "$EPOCHS" ]; then usage "Number of epochs is not specified"; else VALUE_E=$EPOCHS; fi;
if [ -z "$BATCH" ]; then usage "Minibatch size is not specified"; else VALUE_B=$BATCH; fi;
if [ -z "$PROCESSING" ]; then usage "Processing information is not specified"; else VALUE_P=$PROCESSING; fi;
if [ -z "$IMBALANCE" ]; then usage "No imbalance strategy is specified"; else VALUE_I=$IMBALANCE; fi;
if [ -z "$METRIC" ]; then usage "Evaluation metric is not specified"; else VALUE_M=$METRIC; fi;
if [ -z "$KFOLD" ]; then usage "Number of kfolds is not specified"; else VALUE_K=$KFOLD; fi;


#Asserting the directory and output subdirectory exists
if [ ! -d ${PWD%/*}/$IN_DIR ]; then
    echo "The specified directory name '${IN_DIR}' does not exist in ${PWD%/*}"
    exit 1
elif [ ! -e ${PWD%/*}/$IN_DIR/output ]; then
    mkdir -p ${PWD%/*}/$IN_DIR/output
fi

#Passing the processing of the data information correctly
if [ "$VALUE_P" == "none" ]; then
    VALUE_N=false;
    VALUE_D=false;
elif [ "$VALUE_P" == "norm" ]; then
    VALUE_N=true;
    VALUE_D=false;
elif [ "$VALUE_P" == "denoise" ]; then
    VALUE_N=false;
    VALUE_D=true;
else
    VALUE_N=true;
    VALUE_D=true;
fi;

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
        --batch $VALUE_B \
        --weight $WEIGHTS \
        --sampling $SAMPLING \
        --metric $VALUE_M \
        --kfold $VALUE_K
