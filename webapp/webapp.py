import numpy as np
import pandas as pd
import cv2
import cPickle as pickle
from PIL import Image
from latex_to_code import generate_latex
from model import build_model
from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    file_input = request.files['input1']
    img = Image.open(file_input)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    latex_code = generate_latex(mdl, img, label_dict)
    return latex_code


if __name__ == '__main__':
    mdl = build_model()
    mdl.initialize()
    mdl.load_params_from('models/cnn_final_nonoise.pkl')
    labels_df = pd.read_csv('data/images/labels.csv')
    label_dict = dict(zip(labels_df['encode'], labels_df['label']))

    app.run(host='0.0.0.0', port=8080, debug=True)
