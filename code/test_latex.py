from IPython.lib.latextools import latex_to_png
import matplotlib.pyplot as plt
from latex_to_code import *
import cPickle as pickle
import cv2


def create_latex(filename, eq, fontsize=50):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, '$%s$' % eq, fontsize=fontsize,
            ha='center', va='center')
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)


if __name__ == '__main__':
    filename = 'test.png'
    eq = r'\alpha+\beta'
    print 'Creating File...'
    create_latex(filename, eq)
    print 'Reading Image...'
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print 'Finding symbols...'
    rects = find_symbols(img)
    rects.sort(key=lambda x: x[0])      # sort by xcoord

    print 'Loading model...'
    with open('models/cnn.pkl', 'r') as f:
        mdl = pickle.load(f)
    labels_df = pd.read_csv('data/images/labels.csv')
    label_dict = dict(zip(labels_df['encode'], labels_df['labels']))
    for rect in rects:
        predict_symbol(mdl, img, rect, label_dict, show_rect=True)

    print generate_latex(mdl, img, rects, label_dict)
