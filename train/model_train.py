import cv2
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import numpy as np
import argparse

# classes of each output
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "E", "H", "I",
            "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y", "Z"]

WIDTH = 28
HEIGHT = 28
NUM_CLASSES = 36

train_data = np.load("../data/tensors/train_data.npy")
train_labels = np.load("../data/tensors/train_labels.npy")
valid_data = np.load("../data/tensors/valid_data.npy")
valid_labels = np.load("../data/tensors/valid_labels.npy")
test_data = np.load("../data/tensors/test_data.npy")
test_labels = np.load("../data/tensors/test_labels.npy")

train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
test_labels = to_categorical(test_labels)

save_pb_dir = "."

def freeze_graph(graph, session, output, save_pb_dir=save_pb_dir,
                save_pb_name="model.pb", save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session,
                                                            graphdef_inf,
                                                            output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name
                            as_text=save_pb_as_text)
        return graphdef_frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, help="file name to save model", default=None, required=True)
    parser.add_argument("--epochs", "-e", type=int, help="number of epochs", default=12)
    parser.add_argument("--batch", "-b", type=int, help="batch size", default=32)

    args = parser.parse_args()

    model = Sequential()
    # input: 28x28 images with 1 channel -> (28, 28, 1) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation="relu",
                input_shape=(WIDTH, HEIGHT, 1)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy",
                    metrics=["accuracy"])
    model.summary()

    history = model.fit(train_data, train_labels,
              validation_data=(valid_data, valid_labels),
              epochs=args.epochs, batch_size=args.batch, verbose=1)

    predictions = [np.argmax(model.predict(np.expand_dims(tensor,
                                            axis=0))) for tensor in test_data]


    test_accuracy = 100*np.sum(np.array(predictions) == np.argmax(test_labels,
                                                        axis=1))/len(predictions)
    print("Test accuracy: %.1f%%" % test_accuracy)

    # Save the h5 file
    model.save(args.file)
    print("model saved to current working directory as %s", % args.file)


    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)

    model = load_model(args.file)
    session = tf.keras.backend.get_session()

    input_names = [t.op.name for t in model.inputs]
    output_names = [t.op.name for t in model.outputs]

    print(input_names, output_names)

    frozen_graph = freeze_graph(session.graph, session,
                                [out.op.name for out in model.outputs],
                                save_pb_dir=save_pb_dir)

    trt_graph = trt.create_inference_graph(
                    input_graph_def=frozen_graph,
                    outputs=output_names,
                    max_batch_size=1,
                    max_workspace_size_bytes=1<<25,
                    precision_mode="FP16",
                    minimum_segment_size=50)

    graph_io.write_graph(trt_graph, ".", "model.pb", as_text=False)




