import sys
import os
import cv2
import numpy as np
import pandas as pd
from string import ascii_lowercase, ascii_uppercase
from IPython.lib.latextools import latex_to_png
import matplotlib.pyplot as plt


def generate_images(path, values, fontsize=50, figsize=(5,5)):
    '''
    Converts a list of values to latex images and saves them to specified
    path.

    input: path - (string) Specified path for images to be saved
           values - (list) List of values to be converted to latex images
           fontsize - (int) Font size for the symbol
    '''
    for val in values:
        fig, ax = plt.subplots(figsize=figsize)
        # minus sign work around
        if val == '-':
            txt = '_'
        else:
            txt = '$%s$' % val
            
        ax.text(0.5, 0.5, txt, fontsize=fontsize,
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


def img_to_array(path, n_noise=100):
    '''
    Converts all images in path to a 28x28 matrix representation of the
    image and creates new images with noise in them.

    input: path - (string) Folder containing images to be converted
           n_noise - (int) Number of noise images to add. If 0, then just the
                     original image without noise will be saved.
    '''
    img_dict = {'label': [], 'img': []}
    for f in os.listdir(path):
        img = get_img(path + '/' + f)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        label = f.split('_')[0]
        if n_noise > 0:
            for _ in xrange(n_noise):
                img_dict['label'].append(label)
                noise_img = add_noise(img)
                img_dict['img'].append(noise_img)
        else:
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
                print label
                print f
                return
            else:
                label_dir[label] = count
                count += 1

        df['label_encode'] = df['label'].map(label_dir)
        res = pd.concat((res, df), ignore_index=True)

    res.to_json('data/images/compiled.json')
    label_str = map(lambda x: str(x), label_dir.keys())
    label_str = '\n'.join(label_str)
    with open('data/images/labels.txt', 'w') as f:
        f.write(label_str)

    label_df = pd.DataFrame({'labels': label_dir.keys(),
                             'encode': label_dir.values()})
    label_df.to_csv('data/images/labels.csv', index=False)


def add_noise(img):
    '''
    Adds noise to inputted image.

    input: img - A numpy array representing an image.
    output: (numpy array) - New image with noise added.
    '''
    noise = np.zeros(img.shape, dtype=np.uint8)
    cv2.randn(noise, 0, 150)
    new_img = img + noise
    return new_img


def rm_images():
    '''
    Removes all images before running the program.
    '''
    paths = ['imgs/numbers/',
             'imgs/letters/lower/',
             'imgs/letters/upper/',
             'imgs/letters/greek_lower/',
             'imgs/letters/greek_upper/',
             'imgs/operators/',
             'data/images/']
    for path in paths:
        for f in os.listdir(path):
            os.remove(path + f)


def main():
    print 'Removing images...'
    rm_images()
    print 'Generating symbol images...'
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

    if len(sys.argv) > 1:
        n_noise = int(sys.argv[1])
    else:
        n_noise = 0

    print 'Converting images to matrices...'
    img_to_array('imgs/numbers', n_noise)
    img_to_array('imgs/letters/lower', n_noise)
    img_to_array('imgs/letters/upper', n_noise)
    img_to_array('imgs/letters/greek_lower', n_noise)
    img_to_array('imgs/letters/greek_upper', n_noise)
    img_to_array('imgs/operators', n_noise)

    print 'Compiling images to one dataset...'
    compile_images()


if __name__ == '__main__':
    main()
