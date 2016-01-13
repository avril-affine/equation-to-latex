from IPython.lib.latextools import latex_to_png
from model import build_model
import matplotlib.pyplot as plt
# from latex_to_code import *
from Latex2Code import Latex2Code
import cPickle as pickle
import pandas as pd
import cv2


def create_latex(filename, eq, fontsize=50, figsize=(5,5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, '$%s$' % eq, fontsize=fontsize,
            ha='center', va='center')
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)


if __name__ == '__main__':
    filename = 'test.png'
    # eq = r'\frac{x+y}{z+w}'
    # eq = r'\frac{\frac{A}{\beta}+\gamma}{x+y}=z'
    # eq = r'x-y'
    # eq = r'a+b=c'
    # eq = r'81 \leq x \geq 192=-1'
    # eq = r'\int \frac{1}{x}dx'
    eq = r'yA_{Ay}^{xy}'
    # eq = r'\sum_{i=0}^{2} x^{2}'
    # eq = r'(X^{T}X)^{-1}X^{T}y'
    print 'Creating File...'
    create_latex(filename, eq, fontsize=100, figsize=(10,10))
    print 'Reading Image...'
    img = cv2.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # print 'Finding symbols...'
    # rects = find_symbols(img)
    # rects.sort(key=lambda x: x[0])      # sort by xcoord

    print 'Loading model...'
    labels_df = pd.read_csv('data/images/labels.csv')

    num_labels = len(labels_df['label'].unique())
    mdl = build_model(num_labels)
    mdl.initialize()
    mdl.load_params_from('models/cnn_handle_frac_last.pkl')
#    for rect in rects:
#        predict_symbol(mdl, img, rect, label_dict, show_rect=True)

    latex = Latex2Code(mdl, labels_df, verbose=True)
    print latex.to_latex(img)
    latex.cropImage()
    # print generate_latex(mdl, img, label_dict)
