# Scripts for image preprocessing

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

* for `training_data/`, `./search.sh ../data/raw/training_data`: 6286 images
* for `test_data/`: 360 images
* for `valiation_data/`: 545 images

`excheck.sh` finds if there are any non-jpeg or non-jpg files present in the directories. Seemed like I had some strays.
