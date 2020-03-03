import cv2
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l2
from keras.models import Model
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# classes of each output
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I",
            "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y", "Z"]

WIDTH = 28
HEIGHT = 28
NUM_CLASSES = 36
INPUT_SHAPE = (28,28,1)
EPOCHS=8
BATCH_SIZE=32

# training data
train_data = np.load("../data/nparr/train_data.npy")
train_labels = np.load("../data/nparr/train_labels.npy")
# validation data
valid_data = np.load("../data/nparr/valid_data.npy")
valid_labels = np.load("../data/nparr/valid_labels.npy")
# final testing data
test_data = np.load("../data/nparr/test_data.npy")
test_labels = np.load("../data/nparr/test_labels.npy")

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
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name,
                            as_text=save_pb_as_text)
        return graphdef_frozen


def resnet_layer(inputs,
                num_filters=14,
                kernel_size=3,
                strides=1,
                activation='relu',
                batch_normalization=True,
                conv_first=True):
    """
    2D convolution batch-normalization-activation stack builder
    Arguments
        inputs: input tensor from input image or prev layer
        num_filters: conv2D number of filters
        kernel_size: conv2D square kernel dimensions
        strides: conv2D square stride dimensions
        activation: activation name
        batch_normalization: to include batch_normalization or not
        conv_first: order of convolution
    Returns
        x: tensor as input to next layer
    """

    conv = Conv2D(num_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))

    x = inputs

    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)

    return x

def resnet_model(input_shape, depth, num_classes):
    """
    resnet model builder
    Arguments
        input_shape: shape of the input image tensor
        depth: number of core convolutional layers
        num_classes: number of classes
    Returns
        model: Keras model instance
    """

    num_filters = 14
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # for first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                            num_filters=num_filters,
                            strides=strides)
            y = resnet_layer(inputs=y,
                            num_filters=num_filters,
                            activation=None)
            if stack > 0 and res_block == 0:  # for first layer but not first stack
                # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x,
                                num_filters=num_filters,
                                kernel_size=1,
                                strides=strides,
                                activation=None,
                                batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # add classifier on top
    x = AveragePooling2D(pool_size=(2,2))(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # instantiate model
    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    if args.train:
        print("Building model...")

        model = resnet_model(input_shape=INPUT_SHAPE, depth=20,
                            num_classes=NUM_CLASSES)
        model.compile(loss="categorical_crossentropy",
                    optimizer="Adam", metrics=["accuracy"])

        # training
        print("Fitting model...")
        model.fit(train_data, train_labels,
                validation_data=(valid_data, valid_labels),
                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                shuffle=True)


        # testing
        label_pred = model.predict(test_data)
        test_accuracy = accuracy_score(test_labels, label_pred.round())
        print("Test acurracy", test_accuracy)

        # Save the h5 file
        model.save("model.h5")
        print("model saved to current working directory")

    elif args.export:
        print("Exporting model...")
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
