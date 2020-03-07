#!/bin/sh
# sum amount of files in the directory

cd $1

for d in */; do
    cd "$d"
    val=$(ls | wc -l)
    sum=$((sum+val))
    cd ../
done
echo $sum
