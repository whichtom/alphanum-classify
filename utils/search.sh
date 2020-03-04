#!/usr/bin/env bash
# sum amount of files in the directory

cd $1

for d in */; do
    cd "$d"
    val=$(ls -l | wc -l)
    sum=$((sum+val))
    cd ../
done
echo $sum
