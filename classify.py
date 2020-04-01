import numpy as np
import keras
from keras.models import Sequential, load_model
import cv2
import sys

CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I",
            "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y", "Z"]

def preprocess_img(img_path):
    # this is a super janky way to do it and needs to be cleaned up
    img = cv2.imread(img_path, 0)
    img = img.flatten()
    img = img.astype(int)
    img = np.array([np.reshape(img, (28, 28))])
    img = np.array([img.flatten()])
    img = img.astype("float32") / 255
    img = img.reshape(img.shape[0], 28, 28, 1)
    return img

def classify(img_path):

    # preprocess the img
    print("preprocessing image")
    img = preprocess_img(img_path)

    # predict the class from the img
    print("predicting class")
    class_pred_idx = model.predict_classes(img)
    assert len(class_pred_idx) == 1

    class_pred = CLASSES[class_pred_idx[0]]
    return class_pred


if __name__=="__main__":
    if len(sys.argv) < 2:
        print("use: $ python3 {} <path/to/image.jpg>".format(sys.argv[0]))
        exit(-1)

    model = load_model("train/model.h5")

    class_pred = classify(img_path=sys.argv[1])
    print("predicted alphanumeric character: {}".format(class_pred))
