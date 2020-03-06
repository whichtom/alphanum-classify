"""
argv[1] is the directory of the raw image data, argv[2] is the directory where the .csv are saved
"""
import os
import sys
import numpy as np
from create_csv import img_to_csv

train_data, train_labels = img_to_csv(sys.argv[1] + "/training_data")
valid_data, valid_labels = img_to_csv(sys.argv[1] + "/validation_data")
test_data, test_labels = img_to_csv(sys.argv[1] + "/test_data")

print("Saving training data...")
np.savetxt(sys.argv[2] + "/train_data.csv", train_data, delimiter=",", fmt="%d")
np.savetxt(sys.argv[2] + "/train_labels.csv", train_labels, delimiter=",", fmt="%d")
print("Saving validation data...")
np.savetxt(sys.argv[2] + "/valid_data.csv", valid_data, delimiter=",", fmt="%d")
np.savetxt(sys.argv[2] + "/valid_labels.csv", valid_labels, delimiter=",", fmt="%d")
print("Saving test data...")
np.savetxt(sys.argv[2] + "/test_data.csv", test_data, delimiter=",", fmt="%d")
np.savetxt(sys.argv[2] + "/test_labels.csv", test_labels, delimiter=",", fmt="%d")
