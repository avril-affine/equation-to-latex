from IPython.lib.latextools import latex_to_png
from model import build_model
import matplotlib.pyplot as plt
# from latex_to_code import *
from Latex2Code import Latex2Code
import cPickle as pickle
import pandas as pd
import cv2
import os


def main():
    print 'Loading model...'
    labels_df = pd.read_csv('data/images/labels.csv')
    num_labels = len(labels_df['label'].unique())

    mdl = build_model(num_labels)
    mdl.initialize()
    mdl.load_params_from('models/cnn_handle_frac_last.pkl')

    latex = Latex2Code(mdl, labels_df, verbose=True)

    test_df = pd.read_csv('test_imgs.csv')

    preds = []
    for filename in test_df['filename']:
        print 'Predicting', filename
        img = cv2.imread('test_imgs/' + filename)
        preds.append(latex.to_latex(img))

    y_true = [s.replace(' ', '') for s in test_df['equation']]
    preds = [s.replace(' ', '') for s in preds]

    y_true = np.array(y_true)
    preds = np.array(preds)

    print 'Accuracy =', 1. * sum(y_true == preds) / len(y_true)


if __name__ == '__main__':
    main()
