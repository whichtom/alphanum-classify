"""
Preprocessing for converting raw images to csv files with labels.
"""

import os
import sys
import cv2
import numpy as np
import csv
import re
import glob
#from keras.preprocessing.image import img_to_array

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
        img: image from the directory
    """

    imgs = []
    for filename in os.listdir(folderdir):
        img = cv2.imread(os.path.join(folderdir, filename), 0)
        if img is not None:
            imgs.append(img)
#            if filename == "1415.jpg":
#                cv2.imshow("image",img)
#                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
#                cv2.resizeWindow("image", 300, 300)
#
#                cv2.waitKey(0)
    return imgs

def get_subdir(folderdir):
    dirlist = [x[0] for x in os.walk(folderdir)]
    dirlist = dirlist[1:len(dirlist)]
    return dirlist

def img_to_csv(imgdir):
    """
    Converts a list of images to a csv for ease of use in ML
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

    data = np.zeros([num_files, 785]) # 28 x 28 + 1
    label = np.zeros([num_files])

    dirlist = get_subdir(imgdir)
    dirlist.sort(key=natural_keys) # sorts
    k = 0 # per image 
    p = 0 # per class, index 0-35
    for i in dirlist:
        imgs = load_img_from_folder(i)
        for j in imgs:
            # individual img is j
            # individual label is generated for said img
            j = j.flatten()
            j = j.astype(int)
            data[k, 0:len(j)] = j
            data[k, len(j)] = int(p)
            label[k] = int(p)
            k = k+1
        p = p+1

    return data, label

if __name__=="__main__":
    test_data, test_label = img_to_csv("../data/raw/test_data")
    np.savetxt("test.csv", test_data, delimiter=",", fmt="%d")
