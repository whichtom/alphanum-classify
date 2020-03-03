# UAS Reconnaissance Software

The goal is to develop a system capable of recognition of alphanumeric characters on ground targets, classify, locate with lat/lon, and downlink to a GCS from a UAS.

atm just does image recognition for a dataset of alphanumeric characters.

## Usage

### Building npy dataset from raw

```
$ python3 save_nparray.py ../data/raw/ ../data/nparr/
```

Where `../data/raw/` is the directory for the raw images, and `../data/nparr/` is the target directory of the npy files. The labels produced are indices for the CLASSES list (0-35), and so can be easily decipherd.

### Usage

```
$ python3 model_train.py --train
$ python3 model_train.py --export
```
