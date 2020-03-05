# UAS Reconnaissance Software

The goal is to develop a system capable of recognition of alphanumeric characters on ground targets, classify, locate with lat/lon, and downlink to a GCS from a UAS.

atm just does image recognition for a dataset of alphanumeric characters.

## Usage

### Building npy dataset from raw

```bash
$ ./rawtonpy.sh
```
Will run the python script `save_nparray.py`, converting raw images to .npy files with labels. If you want to overwrite current files already present in the `/data/nparr/` directory simply
```bash
$ ./rawtonpy.sh overwrite
```

Where `/data/raw/` is the directory for the raw images, and `/data/nparr/` is the target directory of the npy files. This script loads the raw images, and uses the `create_nparray.py` scriptto process the raw data and generate labels, saving into designated variables as shown below in the excerpt from `save_nparray.py`.

```python
train_data, train_labels = img_to_nparray(sys.argv[1] + "/training_data")
valid_data, valid_labels = img_to_nparray(sys.argv[1] + "/validation_data")
test_data, test_labels = img_to_nparray(sys.argv[1] + "/test_data")
```

The `create_nparray.py` script assigns labels from knowledge of how the raw data is stored. In the directory `data/raw/` there are three subdirectories, `/training_data/`, `/validation_data/`, and `/test_data/`. Within each of these are separate folders for each alphanumeric character (0-9, A-Z). The script walks through the directories, applies `cv2.COLOR_BGR2GRAY`, `keras.preprocessing.image.img_to_array`, normalizes between 0 and 1, and then assigns labels depending on an iterator. This iterator is used as a index for a CLASSES list, producing type `str` values for labels.

Ultimately this produces `.npy` files in the `sys.argv[2]` directory.

### Training and exporting 

Training the model is simple once the data is in the correct format. Simply,

```
$ python3 model_train.py --train
```

