<p align="center"><img src="https://raw.githubusercontent.com/whichtom/reconnaissance/master/assets/banner.png"/></p>

# Alphanumeric character classification

The goal is to develop a system capable of recognition of alphanumeric characters on ground targets, classify, locate with lat/lon, and downlink to a GCS from a UAS.

For now this is just character classification.

## Results

The model trained with the dataset is at 97% test accuracy, using a batch size of 16 for 64 epochs.

<p align="center"><img src="https://raw.githubusercontent.com/whichtom/reconnaissance/master/assets/training-val-acc.png"/></p>
<p align="center"><img src="https://raw.githubusercontent.com/whichtom/reconnaissance/master/assets/training-val-loss.png" /></p>

## Requirements

These have been tested on a NVIDIA Jetson Nano JetPack 4.3 - L4T R32.3.1

* Ubutu 18.04.4 LTS
* opencv
* tensorflow 1.15.0
* keras
* sklearn
* matplotlib
* numpy

For the Jetson Nano, sklearn was installed through apt-get, most else through pip. OpenCV was built from source like so.

```
$ dependencies=(build-essential
              cmake
              pkg-config
              libavcodec-dev
              libavformat-dev
              libswscale-dev
              libv4l-dev
              libxvidcore-dev
              libavresample-dev
              python3-dev
              libtbb2
              libtbb-dev
              libtiff-dev
              libjpeg-dev
              libpng-dev
              libtiff-dev
              libdc1394-22-dev
              libgtk-3-dev
              libcanberra-gtk3-module
              libatlas-base-dev
              gfortran
              wget
              unzip)
$ sudo apt install -y ${dependencies[@]}
$ wget https://github.com/opencv/opencv/archive/4.2.0.zip -O opencv-4.2.0.zip
$ wget https://github.com/opencv/opencv_contrib/archive/4.2.0.zip -O opencv_contrib-4.2.0.zip
$ unzip opencv-4.2.0.zip 
$ unzip opencv_contrib-4.2.0.zip
$ mkdir opencv-4.2.0/build 
$ cd opencv-4.2.0/build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_PTX="" \
      -D CUDA_ARCH_BIN="5.3,6.2,7.2" \
      -D WITH_CUBLAS=ON \
      -D WITH_LIBV4L=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_java=OFF \
      -D WITH_GSTREAMER=OFF \
      -D WITH_GTK=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.2.0/modules \
      ..
$ make -j4
$ sudo make install
``` 

Tensorflow was installed like so.
```
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v4.3 tensorflow==1.15.0+nv20.01
```

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

I didn't manage to achieve very high accuracy using .npy, so maybe something weird was happening to the data. 

## Training the model

Training the model is simple once the data is in the correct format. Run the script `model_train.py --train` in the train directory with the flags:
* `--train` trains the model
* `-e` or `--epoch` followed by an int, specifying the number of epochs to train for. Default 64
* `-b` or `--batch` followed by an int, specifying the batch size. Default 32.
* `-f` or `--file` followed by a str, specifying the file name that the model will be saved to as a .h5, without optimizer (to fix memory leaking). Default "model".

Recommended running arguments are:

```
$ python3 model_train.py --train -e 64 -b 32 -f model
```

Might get some accuracy increase with epoch increase, but not tested above 128. Batch of 64 might create memory issues, so 32 or 16 is good.

## Classifying

Run the `classify.py` script with the argument being the .jpg to be classified. For example,

```
$ python3 classify.py data/raw/test_data/A/449.jpg
```
Which returns the predicted alphanumeric character. Supports 28x28 images.

## To do

The system built does very basic image classification, but it seems I'll need to rebuild the dataset to train an object detection system to better perform the task of reconnaissance. yay.
