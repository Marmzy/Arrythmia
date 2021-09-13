#!/usr/bin/env bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage(){
    if [ -n "$1" ]; then
        echo -e "${RED}→ $1${CLEAR}"
    fi
    echo "Usage: $0 [-h help] [-o output]"
    echo " -h, --help       Print this help and exit"
    echo " -d, --data       Name of the data directory"
    echo " -l, --layers     Number of layers for the ResNet model"
    echo " -r, --lr         ADAM learning rate"
    echo " -e, --epochs     Number of epochs"
    echo ""
    echo "Example: $0 -d data -l 34, -r 0.0001 -e 25"
    exit 1
}

#Parsing command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -d|--data) INPUT="$2"; shift ;;
        -l|--layers) LAYERS="$2"; shift ;;
        -r|--lr) LR="$2"; shift ;;
        -e|--epochs) EPOCHS="$2"; shift ;;
    esac
    shift
done

#Verifying arguments
if [ -z "$INPUT" ]; then usage "Input directory name is not specified" ; else IN_DIR=${INPUT}; fi;
if [ -z "$LAYERS" ]; then usage "Number of ResNet layers is not specified"; else VALUE_L=$LAYERS; fi
if [ -z "$LR" ]; then usage "Learning rate if not specified"; else VALUE_R=$LR; fi
if [ -z "$EPOCHS" ]; then usage "Number of epochs is not specified"; else VALUE_E=$EPOCHS; fi;

#Asserting the 
if [ ! -d ${PWD%/*}/$IN_DIR ]; then
    echo "The specified directory name '${IN_DIR}' does not exist in ${PWD%/*}"
    exit 1
fi

#Training the model
python3 ${PWD%/*}/src/data_train.py \
        --data $IN_DIR \
        --layers $VALUE_L \
        --lr $VALUE_R \
        --epochs $VALUE_E
