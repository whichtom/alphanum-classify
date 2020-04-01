import cv2
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical, plot_model
from keras import backend as K
from keras.regularizers import l2
from keras.models import Model
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import datetime

# classes of each output
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I",
            "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y", "Z"]

WIDTH = 28
HEIGHT = 28
NUM_CLASSES = 36
INPUT_SHAPE = (WIDTH, HEIGHT, 1)

# training data
train_data = np.loadtxt("../data/csv/train_data.csv", delimiter=",")
train_labels = np.loadtxt("../data/csv/train_labels.csv", delimiter=",")
# validation data
valid_data = np.loadtxt("../data/csv/valid_data.csv", delimiter=",")
valid_labels = np.loadtxt("../data/csv/valid_labels.csv", delimiter=",")
# final testing data
test_data = np.loadtxt("../data/csv/test_data.csv", delimiter=",")
test_labels = np.loadtxt("../data/csv/test_labels.csv", delimiter=",")

# updated preprocessing

train_data = np.array([np.reshape(i, (28, 28)) for i in train_data])
train_data = np.array([i.flatten() for i in train_data])

valid_data = np.array([np.reshape(i, (28, 28)) for i in valid_data])
valid_data = np.array([i.flatten() for i in valid_data])

test_data = np.array([np.reshape(i, (28, 28)) for i in test_data])
test_data = np.array([i.flatten() for i in test_data])

label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)
valid_labels = label_binarizer.fit_transform(valid_labels)
test_labels = label_binarizer.fit_transform(test_labels)

train_data = train_data.astype("float32") / 255
valid_data = valid_data.astype("float32") / 255
test_data = test_data.astype("float32") / 255

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
valid_data = valid_data.reshape(valid_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epoch", "-e", type=int, help="epoch", default=64)
    parser.add_argument("--batch", "-b", type=int, help="batch size", default=32)
    parser.add_argument("--file", "-f", type=str, help="file name to save model to", default="model")
    args = parser.parse_args()

    if args.train:
        print("="*80)
        print("Building model...")
        print("="*80)
        model = Sequential()
        # input: 28x28 images with 1 channel -> (28, 28, 1) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        model.add(Conv2D(32, (3, 3),
                    input_shape=(WIDTH, HEIGHT, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.50))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.50))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(rate=0.5))
        model.add(Dense(NUM_CLASSES))
        model.add(Activation("softmax"))

        model.compile(optimizer=RMSprop(learning_rate=1e-3,
                        decay=1e-6),
                        loss="categorical_crossentropy",
                        metrics=["accuracy"])

        # visualization
        model.summary()
        print("="*80)
        print("== Visualization ==")
        print("Saving model with keras.utils.plot_model as model_arch.png")
        plot_model(model, to_file="model_arch.png")

        # training
        print("Fitting model...")
        print("="*80)
        history = model.fit(train_data, train_labels,
                validation_data=(valid_data, valid_labels),
                epochs=args.epoch, batch_size=args.batch, verbose=1)

        # testing
        label_pred = model.predict(test_data)
        test_accuracy = accuracy_score(test_labels, label_pred.round())
        print("Test acurracy: {}".format(test_accuracy))

        # Save the h5 file
        model.save("{}.h5".format(args.file), include_optimizer=False)
        print("="*80)
        print("model saved to current working directory")

        # matplotlib visualization
        # training and validation accuracy values
        print("Plotting training and validation accuracy values...")
        plt.figure(0)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.savefig("training-val-acc.png")

        # training and validation loss values
        print("Plotting training and validation loss values...")
        print("="*80)
        plt.figure(1)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.savefig("training-val-loss.png")
