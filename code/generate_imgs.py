import os
import cv2
import numpy as np
import pandas as pd
from string import ascii_lowercase, ascii_uppercase
from IPython.lib.latextools import latex_to_png
import matplotlib.pyplot as plt


def generate_images(path, values, fontsize=35):
    '''
    Converts a list of values to latex images and saves them to specified
    path.

    input: path - (string) Specified path for images to be saved
           values - (list) List of values to be converted to latex images
           fontsize - (int) Font size for the symbol
    '''
    for val in values:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '$%s$' % val, fontsize=fontsize,
                ha='center', va='center')
        ax.axis('off')
        plt.savefig(path + str(val) + '_' + str(fontsize) + '.png')
        plt.close(fig)


def get_img(filename):
    '''
    Crops an image out of a file leaving only the symbol

    input: filename: (string) path to the image file
    '''
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret, img_thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)

    contours, hier = cv2.findContours(img_thresh.copy(), cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    rect = map(cv2.boundingRect, contours)
    rect = rect[-1]
    cropped = img_gray[rect[1]:(rect[1] + rect[3]),
                       rect[0]:(rect[0] + rect[2])]

    # pt1 = int(rect[1] + rect[3] // 2)
    # pt2 = int(rect[0] + rect[2] // 2)
    # cropped = img[pt1:pt1+rect[3], pt2:pt2+rect[2]]
    # cropped = img[rect[0]:rect[3], rect[1]:rect[2]]
    # return img, rect, cropped
    return cropped


def test_get_img(filename):
    cropped = get_img(filename)
    plt.imshow(cropped)
    plt.show()


def img_to_array(path):
    '''
    Converts all images in path to a 28x28 matrix representation of the
    image.

    input: path - (string) Folder containing images to be converted
    '''
    img_dict = {'label': [], 'img': []}
    for f in os.listdir(path):
        print f
        img = get_img(path + '/' + f)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        label = f.split('_')[0]
        img_dict['label'].append(label)
        img_dict['img'].append(img)

    filename = path.split('/')[-1]
    df = pd.DataFrame(img_dict)
    df.to_json('data/images/' + filename + '.json')


def compile_images():
    '''
    Compiles a folder of json files to a single json and converts labels
    to appropriate ints
    '''
    res = pd.DataFrame({'label': [], 'img': []})
    label_dir = {}
    count = 0
    for f in os.listdir('data/images'):
        df = pd.read_json('data/images/' + f)
        labels = df['label'].unique()
        for label in labels:
            if label in label_dir:
                print 'Error: repeat label in two files'
                return
            else:
                label_dir[label] = count
                count += 1

        df['label'] = df['label'].map(label_dir)
        res = pd.concat((res, df))

    res.reindex()
    print res
    res.to_json('data/images/compiled.json')
    label_str = ','.join(label_dir.keys())
    with open('data/images/labels.txt', 'w') as f:
        f.write(label_str)


def add_noise(path, n):
    '''
    Creates multiple new images with noise.

    input: path - The filename with its path for the image to add noise to.
           n - Number of new images to create.
    '''
    img = cv2.imread(path)
    filename = path.split('.png')[0]
    noise = np.zeros(img.shape, dtype=np.uint8)
    for i in xrange(n):
        cv2.randn(noise, 0, 175)
        new_img = img + noise
        with open(filename + '_' + str(i) + '.png', 'w') as f:
            f.write(new_img)


def main():
    # numbers
    generate_images('imgs/numbers/', range(0, 10))
    # ascii lower
    generate_images('imgs/letters/lower/', ascii_lowercase)
    # ascii upper
    generate_images('imgs/letters/upper/', ascii_uppercase)
    # greek lower
    greekletters = pd.read_csv('data/greeklower.csv')['letters']
    generate_images('imgs/letters/greek_lower/', greekletters)
    # greek upper
    greekletters = pd.read_csv('data/greekupper.csv')['letters']
    generate_images('imgs/letters/greek_upper/', greekletters)
    # operators
    operators = pd.read_csv('data/operators.csv')['operators']
    generate_images('imgs/operators/', operators)

    img_to_array('imgs/numbers')
    img_to_array('imgs/letters/lower')
    img_to_array('imgs/letters/upper')
    img_to_array('imgs/letters/greek_lower')
    img_to_array('imgs/letters/greek_upper')
    img_to_array('imgs/operators')

    compile_images()


if __name__ == '__main__':
    main()
