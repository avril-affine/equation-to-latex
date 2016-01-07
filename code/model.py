from lasagne import layers
from lasagne.nonlinearities import rectify, softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import PrintLayerInfo
from nolearn.lasagne import TrainSplit
# from sklearn.metrics import classification_report
import cPickle as pickle
import pandas as pd
import numpy as np


class MyTrainSplit(TrainSplit):
    def __call__(self, X, y, net):
        train_index = []
        test_index = []
        for val in np.unique(y):
            indexes = np.where(y == val)[0]
            length = len(indexes)
            train_index.extend(indexes[int(length * self.eval_size):])
            test_index.extend(indexes[:int(length * self.eval_size)])
        return X[train_index], X[test_index], y[train_index], y[test_index]


class SaveBestModel(object):
    def __init__(self, name):
        self.best = 0.
        self.name = name
        # self.file_num = None


    def __call__(self, nn, train_history):
#        if self.file_num is None:
        # digits = len(str(nn.max_epochs))
        # file_num = '0:0{}d'.format(digits)
        # self.file_num = '{' + file_num + '}'
        score = train_history[-1]['valid_accuracy']
        if score > self.best:
            self.best = score
            # file_string = self.file_num.format(train_history[-1]['epoch'])
            # file_name = self.name + '_' + file_string + '.pkl'
            nn.save_params_to(self.name)


def build_model():
    num_labels = 112
    filter_size1 = 50
    filter_size2 = 100
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
        conv1_num_filters=filter_size1,
        conv1_filter_size=(3, 3),
        conv1_nonlinearity=rectify,
        conv1_pad=1,

        # pool1
        pool1_pool_size=(2, 2),
        pool1_mode='max',

        # conv2
        conv2_num_filters=filter_size2,
        conv2_filter_size=(3, 3),
        conv2_nonlinearity=rectify,
        conv2_pad=1,

        # pool2
        pool2_pool_size=(2, 2),
        pool2_mode='max',

        # hidden1
        hidden1_num_units=1000,
        hidden1_nonlinearity=rectify,

        # hidden2
        hidden2_num_units=200,
        hidden2_nonlinearity=rectify,

        # output
        output_num_units=num_labels,
        output_nonlinearity=softmax,

        # Optimization
        update=nesterov_momentum,
        update_learning_rate=0.007,
        update_momentum=0.6,
        max_epochs=200,

        # Save best model
        on_epoch_finished=[SaveBestModel('cnn_weights_nonoise.pkl')],

        # My train split
        train_split=MyTrainSplit(eval_size=0.2),

        regression=False,
        verbose=2
    )

    return mdl


if __name__ == '__main__':
    df = pd.read_json('compiled.json')
    X = df['img']
    X = list(X.map(lambda x: list(np.array(x, np.float32) / 255)).values)
    X = np.array(X, dtype=np.float32)
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    # y = df['label']
    y = df['encode'].values.astype(np.int32)
    # y = y.reshape(y.shape[0], 1)

    mdl = build_model()

    # layer_info = PrintLayerInfo()
    # mdl.initialize()
    # layer_info(mdl)

    mdl.fit(X, y)
    mdl.save_params_to('cnn_final_nonoise.pkl')
    # train_X, test_X, train_y, test_y = mdl.train_split(X, y, mdl)
    # preds = mdl.predict(test_X)
    # print classification_report(test_y, preds)
