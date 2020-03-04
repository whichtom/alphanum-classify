#!/usr/bin/env bash

cd $1

for d in */; do
    cd "$d"
    echo "in dir $d:"
    find . -type f \( -not -name "*.jpeg" -not -name "*.jpg" \)
    cd ../
done
