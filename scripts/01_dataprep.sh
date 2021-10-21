#!/usr/bin/env bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage(){
    if [ -n "$1" ]; then
        echo -e "${RED}→ $1${CLEAR}"
    fi
    echo "Usage: $0 [-h help] [-v verbose] [-o output] [-n norm] [-d denoise] [-i imbalance] [-k kfold]"
    echo " -h, --help       Print this help and exit"
    echo " -v, --verbose    Print verbose messages"
    echo " -o, --output     Name of output directory where data will be stored"
    echo " -n, --norm       Normalise the dataset"
    echo " -d, --denoise    Denoise the dataset"
    echo " -i, --imbalance  Strategy to alleviate class imbalance (sampling | weights)"
    echo " -k, --kfold      Number of folds to split the training dataset into"
    echo ""
    echo "Example: $0 -o data -n -d -i sampling -k 5"
    exit 1
}

#Parsing command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -v|--verbose) VERBOSE=true ;;
        -o|--output) OUTPUT="$2"; shift ;;
        -n|--norm) NORM=true ;;
        -d|--denoise) DENOISE=true ;;
        -i|--imbalance) IMBALANCE="$2"; shift ;;
        -k|--kfold) KFOLD="$2"; shift ;;
    esac
    shift
done

#Verifying arguments
if [ -z "$OUTPUT" ]; then OUT_DIR="data"; else OUT_DIR=$OUTPUT; fi;
if [ -z "$VERBOSE" ]; then VALUE_V=false; else VALUE_V=true; fi;
if [ -z "$NORM" ]; then VALUE_N=false; else VALUE_N=true; fi;
if [ -z "$DENOISE" ]; then VALUE_D=false; else VALUE_D=true; fi;
if [ -z "$IMBALANCE" ]; then usage "No imbalance strategy is specified"; else VALUE_I=$IMBALANCE; fi;
if [ -z "$KFOLD" ]; then usage "Number of kfolds is not specified"; else VALUE_K=$KFOLD; fi;


#Making the output directory and subsequent subdirectories if it doesn't exist yet
if [ ! -e ${OUT_DIR} ]; then
    mkdir -p ${PWD%/*}/${OUT_DIR}
    mkdir -p ${PWD%/*}/${OUT_DIR}/raw
    mkdir -p ${PWD%/*}/${OUT_DIR}/train
    mkdir -p ${PWD%/*}/${OUT_DIR}/val
    mkdir -p ${PWD%/*}/${OUT_DIR}/test
fi

#Checking if the raw data has been downloaded or not and preprocessing the data accordingly
if [ -z "$(ls -A ${PWD%/*}/${OUT_DIR}/raw)" ]; then
    DOWNLOAD=True
else
    DOWNLOAD=False
fi

#Passing the sampling strategy correctly
if [ $VALUE_I == "sampling" ]; then
    SAMPLING=true
    WEIGHTS=false
else
    SAMPLING=false
    WEIGHTS=true
fi

#Preparing the data
python3 ${PWD%/*}/src/data_prep.py \
        --verbose ${VALUE_V} \
        --output ${OUT_DIR} \
        --test 0.2 \
        --download ${DOWNLOAD} \
        --norm ${VALUE_N} \
        --denoise ${VALUE_D} \
        --weight ${WEIGHTS} \
        --sampling ${SAMPLING} \
        --kfold ${VALUE_K}
