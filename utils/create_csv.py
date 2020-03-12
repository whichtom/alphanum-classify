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
#            if filename == "53.jpg":
#                cv2.imwrite("53_gray.jpg", img)
#            if filename == "yeet_0_3494.jpeg":
#                cv2.imwrite("53_2_gray.jpg", img)
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
        num_files = 1280 # 5837 (culled) # 6286
    elif imgdir == "../data/raw/test_data":
        num_files = 215 # 284 (culled) # 360
    elif imgdir == "../data/raw/validation_data":
        num_files = 305 # 452 (culled) # 545

    data = np.zeros([num_files, 784]) # 28 x 28
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
            j = j.flatten() # (28,28) -> (784,)
            j = j.astype(int) # no long boiz
            data[k, :] = j
            #data[k, len(j)] = int(p) # last column is index
            label[k] = int(p)
            k = k+1
        p = p+1

    return data, label

