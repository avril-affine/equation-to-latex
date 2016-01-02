from lasagne import layers
from lasagne.nonlinearities import rectify, softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import PrintLayerInfo
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


if __name__ == '__main__':
    num_labels = 101         # just for testing
    filter_size1 = 32
    filter_size2 = 64
    mdl = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('pool1', layers.Pool2DLayer),
                ('conv5', layers.Conv2DLayer),
                ('conv6', layers.Conv2DLayer),
                ('conv7', layers.Conv2DLayer),
                ('pool2', layers.Pool2DLayer),
                ('hidden1', layers.DenseLayer),
                ('hidden2', layers.DenseLayer),
                ('output', layers.DenseLayer)],

        # input
        input_shape=(None, 1, 28, 28),

        # conv1
        conv1_num_filters=filter_size1,
        conv1_filter_size=(3, 3),
        conv1_nonlinearity=rectify,
        conv1_pad=1,

        # conv2
        conv2_num_filters=filter_size1,
        conv2_filter_size=(3, 3),
        conv2_nonlinearity=rectify,
        conv2_pad=1,

        # conv3
        conv3_num_filters=filter_size1,
        conv3_filter_size=(3, 3),
        conv3_nonlinearity=rectify,
        conv3_pad=1,

        # conv4
        conv4_num_filters=filter_size1,
        conv4_filter_size=(3, 3),
        conv4_nonlinearity=rectify,
        conv4_pad=1,

        # pool1
        pool1_pool_size=(2, 2),
        pool1_mode='max',   # try average_inc_pad later

        # conv5
        conv5_num_filters=filter_size2,
        conv5_filter_size=(3, 3),
        conv5_nonlinearity=rectify,
        conv5_pad=1,

        # conv6
        conv6_num_filters=filter_size2,
        conv6_filter_size=(3, 3),
        conv6_nonlinearity=rectify,
        conv6_pad=1,

        # conv7
        conv7_num_filters=filter_size2,
        conv7_filter_size=(3, 3),
        conv7_nonlinearity=rectify,
        conv7_pad=1,

        # pool2
        pool2_pool_size=(2, 2),
        pool2_mode='max',   # try average_inc_pad later

        # hidden1
        hidden1_num_units=300,
        hidden1_nonlinearity=rectify,

        # hidden2
        hidden2_num_units=150,
        hidden2_nonlinearity=rectify,

        # output
        output_num_units=num_labels,
        output_nonlinearity=softmax,

        # Optimization
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.7,
        max_epochs=100,

        regression=False,
        verbose=2
    )

    layer_info = PrintLayerInfo()
    mdl.initialize()
    layer_info(mdl)

    # df = pd.read_json('data/images/compiled.json')
    # X = df['img'].values.astype(np.float32) / 255
    # y = df['label']
    # num_labels = len(df['labels'].unique())

    # mdl.train(X, y)
    # train_X, test_X, train_y, test_y = mdl.train_split(train_X, train_y, mdl)
    # preds = mdl.predict(test_X)
    # print classification_report(test_y, preds)
