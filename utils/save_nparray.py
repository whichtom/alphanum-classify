"""
argv[1] is the directory of the raw image data, argv[2] is the directory where the .npy are saved
"""
import os
import sys
import numpy as np
from create_nparray import img_to_nparray

# classes of each output
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I",
            "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y", "Z"]

print("Loading data from ", sys.argv[1])
train_data, train_labels = img_to_nparray(sys.argv[1] + "/training_data")
valid_data, valid_labels = img_to_nparray(sys.argv[1] + "/validation_data")
test_data, test_labels = img_to_nparray(sys.argv[1] + "/test_data")

print("Saving data to ", sys.argv[2])
print("Saving training data...")
np.save(sys.argv[2] + "/train_data.npy", train_data)
np.save(sys.argv[2] + "/train_labels.npy", train_labels)
print("Saving validation data...")
np.save(sys.argv[2] + "/valid_data.npy", valid_data)
np.save(sys.argv[2] + "/valid_labels.npy", valid_labels)
print("Saving test data...")
np.save(sys.argv[2] + "/test_data.npy", test_data)
np.save(sys.argv[2] + "/test_labels.npy", test_labels)
