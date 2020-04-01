# Scripts for data setup

## csv exporting

`rawtocsv.sh` main running file to run the files with preselected arguments.

`create_csv.py` preprocesses data from raw images.

`save_csv.py` saves the csv files from the preprocessed data.

## npy exporting

`rawtonpy.sh` main running file to run the files with preselected arguments.

`create_nparray.py` preprocesses data from raw images.

`save_nparray.py` saves the npy files from the preprocessed data.

## misc

`search.sh <directory>` finds the sum of the files in a directory. Specifically, for checking whether the number of labels produced matches the number of images present.

* for `training_data/`, `./search.sh ../data/raw/training_data`: 1280 images
* for `test_data/`: 215 images
* for `valiation_data/`: 305 images

`excheck.sh` finds if there are any non-jpeg or non-jpg files present in the directories. Seemed like I had some strays.
