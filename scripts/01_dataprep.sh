#!/usr/bin/env bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage(){
    if [ -n "$1" ]; then
        echo -e "${RED}→ $1${CLEAR}"
    fi
    echo "Usage: $0 [-h help] [-o output]"
    echo " -h, --help       Print this help and exit"
    echo " -o, --output     Name of output directory where data will be stored"
    echo " -b, --hbeat      AAMI heartbeat symbols to classify (select 2 from N, S, V, F & Q)"
    echo ""
    echo "Example: $0 -o data -h NV"
    exit 1
}

#Parsing command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -o|--output) OUTPUT="$2"; shift ;;
        -b|--hbeat) HBEAT="$2"; shift ;;
    esac
    shift
done

#Verifying arguments
if [ -z "$OUTPUT" ]; then OUT_DIR="data"; else OUT_DIR=${OUTPUT}; fi;
if [ -z "$HBEAT" ]; then usage "No heartbeats were selected to classify"; fi


#Making the output directory and subsequent subdirectories if it doesn't exist yet
if [ ! -e ${OUT_DIR} ]; then
    mkdir -p ${PWD%/*}/${OUT_DIR}
    mkdir -p ${PWD%/*}/${OUT_DIR}/raw
    mkdir -p ${PWD%/*}/${OUT_DIR}/train
    mkdir -p ${PWD%/*}/${OUT_DIR}/test
fi

#Checking if the raw data has been downloaded or not and preprocessing the data accordingly
if [ -z "$(ls -A ${PWD%/*}/${OUT_DIR}/raw)" ]; then
    python3 ${PWD%/*}/src/data_prep.py \
            --output ${OUT_DIR} \
            --hbeat ${HBEAT} \
            --download
else
    python3 ${PWD%/*}/src/data_prep.py \
            --output ${OUT_DIR} \
            --hbeat ${HBEAT} \
            --no-download
fi
