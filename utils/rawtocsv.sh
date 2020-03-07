#!/bin/bash

if [[ ! -d "../data" && ! -d "../data/csv" ]]; then
    # no data nor data/csv
    echo "no directory data nor csv. Creating both..."
    mkdir -p ../data/csv
elif [[ -d "../data" && ! -d "../data/csv" ]]; then
    # data but no csv
    echo "no directory csv. Creating directory..."
    mkdir ../data/csv
else
    # both dirs exist
    if [[ $1 == "overwrite" ]]; then
        # overwrite existing nparr
        echo "overwriting existing csv directory..."
        rm ../data/csv/*
    else
        echo "preserving existing csv directory..."

    fi
fi

# now the data/csv exists according to users conditions
echo "saving raw images from ../data/raw as csv to ../data/csv..."
python3 save_csv.py ../data/raw ../data/csv
