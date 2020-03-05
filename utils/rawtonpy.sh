#!/bin/bash

if [[ ! -d "../data" && ! -d "../data/nparr" ]]; then
    # no data nor data/nparr
    echo "no directory data nor nparr. Creating both..."
    mkdir -p ../data/nparr
elif [[ -d "../data" && ! -d "../data/nparr" ]]; then
    # data but no nparr
    echo "no directory nparr. Creating directory..."
    mkdir ../data/nparr
else
    # both dirs exist
    if [[ $1 == "overwrite" ]]; then
        # overwrite existing nparr
        echo "overwriting existing nparr directory..."
        rm ../data/nparr/*
    else
        echo "preserving existing nparr directory..."

    fi
fi

# now the data/nparr exists according to users conditions
echo "saving raw images from ../data/raw as npy to ../data/nparr..."
python3 save_nparray.py ../data/raw ../data/nparr
