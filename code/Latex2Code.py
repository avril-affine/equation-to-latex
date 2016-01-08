import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib


class Latex2Code(object):

    def __init__(self, mdl, img, label_dict, verbose=False):
        self.mdl = mdl
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = img_gray
        self.img_orig = img
        self.label_dict = label_dict
        self.verbose = verbose
        self.symbols = []


    def run(self):
        rects = self.find_symbols()
        rects.sort(key=lambda x: x[0])      # sort by x coord
        self.rects = rects
        for i in xrange(len(rects)):
            symbol = self.predict_symbol(i)
            self.symbols.append(symbol)

        # handle formatted stuff
        latex = r''
        format_symbols = tuple()
        i = 0
        # for i, symbol in enumerate(self.symbols):
        while i < len(self.symbols):
            # test for fraction
            if symbol != '-':
                above, below = self._get_frac_rects(self, i)
                if above:
                    numer = generate_latex(above)
                    denom = generate_latex(below)
                    format_symbols += (numer, denom)
                    latex += r'\frac{%s}{%s}'
                    i += len(above) + len(below)    # move past fraction
                else:
                    latex += symbol
            # can test for other stuff here
            else:
                latex += symbol

        # return ' '.join(self.symbols)     # only handles one line
        return latex % format_symbols


    def _generate_frac_latex(self, rect_indexes):
        symbols = map(lambda i: self.symbols[i], rect_indexes)
        return ' '.join(symbols)            # only handles one line


    def dilate(self, ary, N, iterations):
        """Dilate using an NxN '+' sign shape. ary is np.uint8."""
        kernel = np.zeros((N,N), dtype=np.uint8)
        kernel[(N-1)/2,:] = 1
        dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

        kernel = np.zeros((N,N), dtype=np.uint8)
        kernel[:,(N-1)/2] = 1
        dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
        return dilated_image


    def find_symbols(self):
        '''
        Finds all symbols in an image and returns the rectangles that border
        them. Assumes the input is in gray scale.
        '''
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, img_thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        ret, img_thresh = cv2.threshold(self.img, 50, 255,
                                        cv2.THRESH_BINARY_INV)

        img_thresh = self.dilate(img_thresh, 3, 1)

        contours, hier = cv2.findContours(img_thresh.copy(), cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c, h in zip(contours, hier[0]):
            if h[-1] >= 0:
                continue
            rect = cv2.boundingRect(c)
            # rect_img = img.copy()
            # cv2.rectangle(rect_img, (rect[0], rect[1]),
            #               (rect[0] + rect[2], rect[1] + rect[3]),
            #               (0, 255, 0), 3)
            # plt.imshow(rect_img)
            # plt.show()

            rects.append(rect)

        if self.verbose:
            for rect in sorted(rects, key=lambda x: x[0]):
                rect_img = self.img_orig.copy()
                cv2.rectangle(rect_img, (rect[0], rect[1]),
                              (rect[0] + rect[2], rect[1] + rect[3]),
                              (0, 255, 0), 3)
                plt.imshow(rect_img)
                plt.show()
                print ('x0:', rect[0], 'x1:', rect[0] + rect[2],
                       'y0:', rect[1], 'y1:', rect[1] + rect[3])

        return rects


    def predict_symbol(self, rect_index):
        rect = self.rects[rect_index]
        x_pad = 0
        y_pad = 0
        pad_size = 10
        if rect[2] < 15:
            x_pad = pad_size
        if rect[3] < 15:
            y_pad = pad_size
        cropped = self.img[(rect[1] - y_pad):(rect[1] + rect[3] + y_pad),
                           (rect[0] - x_pad):(rect[0] + rect[2] + x_pad)]

        cropped = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
        cropped = cropped.reshape(1, 1, 28, 28)
        pred_index = self.mdl.predict(cropped)[0]
        pred = self.label_dict[pred_index]

        if self.verbose:
            print pred_index, pred
            rect_img = self.img_orig.copy()
            cv2.rectangle(rect_img, (rect[0], rect[1]),
                          (rect[0] + rect[2], rect[1] + rect[3]),
                          (0, 255, 0), 3)
            plt.imshow(rect_img)
            plt.show()

        return pred


    def _get_frac_rects(self, frac_index):
        above = []
        below = []

        frac = self.rects[frac_index]
        for i in xrange(len(self.rects)):
            if i == frac_index:
                continue

            rect = self.rects[i]
            # test x axis
            frac_left = frac[0]
            frac_right = frac[0] + frac[2]
            rect_left = rect[0]
            rect_right = rect[0] + rect[2]

            if rect_left < frac_left or rect_right > frac_right:
                continue

            # test y axis
            frac_y = frac[1]
            rect_y = rect[1]
            if rect_y > frac_y:
                above.append(i)
            else:
                below.append(i)

        return above, below


    def _get_left(self, rect_index, img_rects):
        rect = self.rects[rect_index]
        # find all rects to the left of rect
        rect_leftx = rect[0]
        lefts = []
        for r in img_rects:
            r_rightx = r[0] + r[2]
            if r_rightx < rect_leftx:
                lefts.append(r)

        # get the closest one
        if not lefts:
            closest = lefts[0]
            closest_rightx = closest[0] + closest[2]
            for r in lefts:
                r_rightx = r[0] + r[2]
                if r_rightx > closest_rightx:
                    closest_rightx = r_rightx
                    closest = r

            return closest


    def _is_subscript(self, rect_index, img_rects, labels):
        rect = self.rects[rect_index]
        left = get_left(rect_img_rects)
        mask = map(lambda r: all(left == r), img_rects)
        left_index = np.where(mask)[0][0]
        left_label = labels[left_index]
        if left_label == '-':
            return False


if __name__ == '__main__':
    img = cv2.imread('test4.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = find_symbols(img)

    mdl = joblib.load('models/cnn.pkl')
    symbols = generate_latex(mdl, img, rects)
