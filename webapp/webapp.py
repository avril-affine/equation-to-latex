import numpy as np
import pandas as pd
import cv2
import cPickle as pickle
from flask import Flask, render_template, request
import os
import sys
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', output=None)


@app.route('/submit', methods=['POST'])
def submit():
    file_input = request.files['input1']
    img = np.asarray(bytearray(file_input.read()), dtype=np.uint8)
    img = cv2.imdecode(img, 1)
    latex_code = Latex.to_latex(img)
    return render_template('index.html', output=latex_code)


if __name__ == '__main__':
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.sys.path.insert(1, parent_dir + '/code')
    from Latex2Code import Latex2Code
    from model import build_model

    labels_df = pd.read_csv('../data/images/labels.csv')
    num_labels = len(labels_df['label'].unique())
    mdl = build_model(num_labels)
    mdl.initialize()
    mdl.load_params_from('../models/cnn_handle_frac_last.pkl')
    Latex = Latex2Code(mdl, labels_df)

    app.run(host='0.0.0.0', port=8080, debug=True)
