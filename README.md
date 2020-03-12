<p align="center"><img src="https://raw.githubusercontent.com/whichtom/reconnaissance/master/assets/banner.png"/></p>

# UAS Reconnaissance Software

The goal is to develop a system capable of recognition of alphanumeric characters on ground targets, classify, locate with lat/lon, and downlink to a GCS from a UAS.

Status: just alphanumeric character recognition atm.

## Results

The model trained with the dataset is at 97% test accuracy, using a batch size of 16 for 64 epochs.

<p align="center"><img src="https://raw.githubusercontent.com/whichtom/reconnaissance/master/assets/training-val-acc.png"/></p>
<p align="center"><img src="https://raw.githubusercontent.com/whichtom/reconnaissance/master/assets/training-val-loss.png" /></p>



## Usage

### Building csv dataset from raw

```bash
$ ./rawtocsv.sh
```
Will run the python script `save_csv.py`, converting raw images to .csv with an accompanied .csv for labels. If you want to overwrite current files already present in the `/data/csv` directory simply
```bash
$ ./rawtocsv.sh overwrite
```

Where `/data/raw/` is the directory for the raw images, and `/data/csv/` is the target directory of the csv files. This script loads the raw images, and uses the `create_csv.py` script to process the raw data and generate labels, asving into designated variables.

The `create_csv.py` script assigns labels from knowledge of how the raw data is stored. In the directory `data/raw/` there are three subdirectories, `/training_data/`, `/validation_data/`, and `/training_data/`. Within each of these are separate folders for each alphanumeric character (0-9, A-Z). The script walks through the directories, for each image flattens the 28x28 image, converts to type int, and loads into a single data array. Labels are generated based on an iterator to be decoded later.

Ultimately this produces `.csv` files in the `sys.argv[2]` directory.

### Building npy dataset from raw (not recommended)

```bash
$ ./rawtonpy.sh
```
Will run the python script `save_nparray.py`, converting raw images to .npy files with labels. If you want to overwrite current files already present in the `/data/nparr/` directory simply
```bash
$ ./rawtonpy.sh overwrite
```

Where `/data/raw/` is the directory for the raw images, and `/data/nparr/` is the target directory of the npy files. This script loads the raw images, and uses the `create_nparray.py` script toprocess the raw data and generate labels, saving into designated variables.

The `create_nparray.py` script assigns labels from knowledge of how the raw data is stored. In the directory `data/raw/` there are three subdirectories, `/training_data/`, `/validation_data/`, and `/test_data/`. Within each of these are separate folders for each alphanumeric character (0-9, A-Z). The script walks through the directories, applies `cv2.COLOR_BGR2GRAY`, `keras.preprocessing.image.img_to_array`, normalizes between 0 and 1, and then assigns labels depending on an iterator. This iterator is used as a index for a CLASSES list, producing type str values for labels.

Ultimately this produces `.npy` files in the `sys.argv[2]` directory.

## Training the model

Training the model is simple once the data is in the correct format. Simply run the script `model_train.py` which accepts the following flags:
* either accepts `--train` or `--export`. The former trains the model, the latter exports the model as a .pb.
* `-e` or `--epoch` followed by an int, specifying the number of epochs to train for. Default 8 (just for testing).
* `-b` or `--batch` followed by an int, specifying the batch size. Default 32.

eg.

```
$ python3 model_train.py --train -e 64 -b 64
```


### Exporting the model

WIP



