from lasagne import layers
from lasagne.nonlinearities import rectify, softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.read_json('data/images/compiled.json')
    X = df['img'].values.astype(np.float32) / 255
    y = df['label']
    num_labels = len(df['labels'].unique())

    mdl = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.Pool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.Pool2DLayer),
                ('hidden1', layers.DenseLayer),
                ('hidden2', layers.DenseLayer),
                ('output', layers.DenseLayer)],

        # input
        input_shape=(None, 1, 28, 28),

        # conv1
        conv1_num_filters=25,
        conv1_filter_size=(3, 3),
        conv1_nonlinearity=recitify,

        # pool1
        pool1_pool_size=(2, 2),
        pool1_mode='average_inc_pad',   # mean pool including zeros

        # hidden1
        hidden1_num_units=200,
        hidden_nonlinearity=recitify,

        # conv2
        conv2_num_filters=25,
        conv2_filter_size=(3, 3),
        conv2_nonlinearity=recitify,

        # pool2
        pool2_pool_size=(2, 2),
        pool2_mode='average_inc_pad',   # mean pool including zeros

        # hidden2
        hidden2_num_units=100,
        hidden_nonlinearity=recitify,

        # output
        output_num_units=num_labels,
        output_nonlinearity=softmax,

        # Optimization
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.7,
        max_epochs=100,

        regression=False,
        verbose=True
    )

    mdl.train(X, y)
    train_X, test_X, train_y, test_y = mdl.train_split(train_X, train_y, mdl)
    preds = mdl.predict(test_X)
    print classification_report(test_y, preds)
