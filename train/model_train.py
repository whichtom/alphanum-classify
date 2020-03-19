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

print("TENSORFLOW VERSION: {}".format(tf.__version__))

# classes of each output
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I",
            "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y", "Z"]

WIDTH = 28
HEIGHT = 28
DEPTH = 20
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


save_pb_dir = "."

def freeze_graph(graph, session, output, save_pb_dir=save_pb_dir,
                save_pb_name="model.pb", save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session,
                                                            graphdef_inf,
                                                            output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name,
                            as_text=save_pb_as_text)
        return graphdef_frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--epoch", "-e", type=int, help="epoch", default=8)
    parser.add_argument("--batch", "-b", type=int, help="batch size", default=32)
    parser.add_argument("--file", "-f", type=str, help="filename to save model")
    args = parser.parse_args()

    if args.train:
        print("="*80)
        print("BUILDING MODEL")
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
        print("SAVING MODEL ARCHITECTURE IMAGE TO CURRENT WORKING DIRECTORY AS model_arch.png")
        plot_model(model, to_file="model_arch.png")

        # training
        print("FITTING MODEL")

        log_dir = "logs/fit/{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = model.fit(train_data, train_labels,
                validation_data=(valid_data, valid_labels),
                epochs=args.epoch, batch_size=args.batch,
                callbacks=[tensorboard_callback], verbose=1)

       # testing
        label_pred = model.predict(test_data)
        test_accuracy = accuracy_score(test_labels, label_pred.round())
        print("TEST ACCURACY (1)", test_accuracy)

        predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_data]
        test_acc = 100*np.sum(np.array(predictions) == np.argmax(test_labels, axis=1))/len(predictions)

        print("TEST ACCURACY (2)", test_acc)
        # Save the h5 file
        model.save("{}.h5".format(args.file))
        print("HDF5 MODEL SAVED TO CURRENT WORKING DIRECTORY AS {}.h5".format(args.file))

        # matplotlib visualization
        # training and validation accuracy values
        print("PLOTTING TRAINING AND VALIDATION ACCURACY")
        plt.figure(0)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.savefig("training-val-acc.png")

        # training and validation loss values
        print("PLOTTING TRAINING AND VALIDATION LOSS")
        print("="*80)
        plt.figure(1)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.savefig("training-val-loss.png")

    elif args.export:
        print("EXPORTING MODEL")
        model = load_model("model.h5")
        session = K.get_session()

        input_names = [t.op.name for t in model.inputs]
        output_names = [t.op.name for t in model.outputs]

        print(input_names, output_names)

        frozen_graph = freeze_graph(session.graph, session,
                                    [out.op.name for out in model.outputs],
                                    save_pb_dir=save_pb_dir)
        converter = trt.TrtGraphConverter(
                                input_graph_def=frozen_graph,
                                max_batch_size=1,
                                max_workspace_size_bytes=1<<25,
                                precision_mode="FP16",
                                minimum_segment_size=50)
        trt_graph = converter.convert()

        graph_io.write_graph(trt_graph, ".", "model2.pb", as_text=False)


    else:
        print("Either --train or --export")
