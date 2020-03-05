"""
Preprocessing for converting raw images to npy files with labels.
"""

import os
import sys
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import re
import glob

CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I",
            "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y", "Z"]


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def listfiles_nohidden(path):
    return [x for x in os.listdir(path) if not x.startswith(".")]

def load_img_from_folder(folderdir):
    """
    Loads images from a folder
    Arguments
        folderdir: directory of the folder to load images from
    Returns
        images: list of images from the directory
    """

    imgs = []
    for filename in os.listdir(folderdir):
        img = cv2.imread(os.path.join(folderdir, filename))
        if img is not None:
            imgs.append(img)
    return imgs

def img_to_nparray(imgdir):
    """
    Converts a list of images to a numpy array for ease of use in ML
    Arguments
        imgdir: directory that the images are in
    Returns
        data, label: numpy array of the data, and another of the respective labels
    """

    # get number of files in the directory
    if imgdir == "../data/raw/training_data":
        num_files = 6286
    elif imgdir == "../data/raw/test_data":
        num_files = 360
    elif imgdir == "../data/raw/validation_data":
        num_files = 545

    data = np.zeros([num_files, 28, 28, 1])
    label = np.empty([num_files], dtype=str)

    dirlist = [x[0] for x in os.walk(imgdir)]
    dirlist = dirlist[1:len(dirlist)]
    dirlist.sort(key=natural_keys) # sorts
    k = 0 # per image 
    p = 0 # per class, index 0-35

    for i in dirlist:
        imgs = load_img_from_folder(i)
        for j in imgs:
            img = cv2.cvtColor(j, cv2.COLOR_BGR2GRAY)
            img = img_to_array(img)
            data[k, :, :, :] = img
            data[k, :, :, :] = (data[k, :, :, :]/255.0)
            label[k] = CLASSES[int(p)]
            print(label[k])
            k = k+1 # iterates per image in the subdirectory in imgdir
        p = p+1 # iterates per subdirectory in imgdir

    return data, label


if __name__=="__main__":
    img_to_nparray("../data/raw/training_data")
