from model import build_model
# from latex_to_code import *
from Latex2Code import Latex2Code
import cPickle as pickle
import pandas as pd
import numpy as np
import cv2
import os


def main():
    print 'Loading model...'
    labels_df = pd.read_csv('labels.csv')
    num_labels = len(labels_df['label'].unique())

    mdl = build_model(num_labels)
    mdl.initialize()
    mdl.load_params_from('models/cnn_handle_frac_last.pkl')


    test_df = pd.read_csv('test_imgs.csv')

    preds = []
    for filename in test_df['filename']:
	latex = Latex2Code(mdl, labels_df, verbose=False)
        print 'Predicting', filename
        img = cv2.imread(filename)
        preds.append(latex.to_latex(img))

    y_true = [s.replace(' ', '') for s in test_df['equation']]
    preds_new = [s.replace(' ', '') for s in preds]

    y_true = np.array(y_true)
    preds_new = np.array(preds_new)

    print 'Accuracy =', 1. * sum(y_true == preds_new) / len(y_true)

    test_df['preds'] = preds
    test_df.to_csv('out.csv')


if __name__ == '__main__':
    main()
