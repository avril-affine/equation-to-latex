import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.externals import joblib


class Latex2Code(object):
    '''
    Class to encapsulate all the data needed to convert an image to latex
    code.

    Parameters:
    mdl: (object) The model to use to predict each symbol
    img: (np array) A gray scale image to be converted to latex code
    img_orig: (np array) Original image used to display cropped out
    images for debugging
    label_dict: (dict) A dictionary to convert labels back to their
    string reprentation
    verbose: (boolean) Used for debugging
    symbols: (list) A list of the string representation of the symbols
    rects: (list) A list of rectangles for the bounding box of each symbol
    '''

    def __init__(self, mdl, label_dict, verbose=False):
        '''
        Init method

        '''
        self.mdl = mdl
        self.label_dict = label_dict
        self.verbose = verbose
        self.symbols = []
        self.rects = []


    def to_latex(self, img):
        '''
        Takes an image and returns the latex code string.

        input:
        img: (np array) The image to be converted

        output: (string) Latex code in string form
        '''
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = img_gray
        self.img_orig = img

        rects = self.find_symbols()
        rects.sort(key=lambda x: x[0])      # sort by x coord
        self.rects = rects
        for i in xrange(len(rects)):
            symbol = self.predict_symbol(i)
            self.symbols.append(symbol)

        return self._generate_frac_latex(r'', tuple(),
                                         range(len(self.symbols)))


    def _generate_frac_latex(self, latex_in, format_in, rect_indexes):
        '''
        Function to check for fraction formatting

        input:
        latex_in: (string) The accumulated latex code string
        format_in: (tuple) Tuple of strings to be inserted into latex_in
        rect_indexes: (list) List of rectangles to be considered within the
        function

        output: (string) The string of the latex code
        '''
        latex = latex_in
        format_symbols = format_in
        i = 0
        while i < len(rect_indexes):
            index = rect_indexes[i]
            symbol = self.symbols[index]
            if symbol == '-':
                above, below = \
                    self._get_frac_rects(index, rect_indexes)
                if below:
                    numer = self._generate_frac_latex(latex,
                                                      format_symbols, above)
                    denom = self._generate_frac_latex(latex,
                                                      format_symbols, below)
                    format_symbols += (numer, denom)
                    latex += r'\frac{%s}{%s}'
                    i += len(above) + len(below)
                elif above:         # handle >= <= and =
                                    # assumes bottom - is always first
                    above_symbol = self.symbols[above[0]]
                    if above_symbol == '>':
                        latex += r'\geq '
                    elif above_symbol == '<':
                        latex += r'\leq '
                    elif above_symbol == '-':
                        latex += '='
                    i += 1
                else:
                    latex += symbol
            else:
                latex += symbol
            i += 1

        return latex % format_symbols


    def _get_frac_rects(self, frac_index, line_indexes):
        '''
        Gets the rectangles that are within the bounds of the fraction symbol

        input:
        frac_index: (int) The index for the fraction symbol in the self.rects
        list
        line_indexes: (list) List of indexes to be considered

        output: (2-tuple) Two lists containing indexes for the above and
        below bounding rectangles.
        '''
        above = []
        below = []

        frac = self.rects[frac_index]
        for i in line_indexes:
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
            if rect_y < frac_y:
                above.append(i)
            else:
                below.append(i)

        return above, below


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
        '''
        Crops the inputted symbol out of the image a returns a prediction
        for that symbol.

        input:
        rect_index: (list) A bounding box for the symbol to be predicted

        output: (string) The string of the predicted symbol
        '''
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


    def _get_left(self, rect_index, img_rects):
        '''
        Gets the closest rectangle to the left of the target
        NOT USED YET. FOR SUPER/SUBSCRIPT!!!!!!!!!!!!!!!!!!!!!!!!!

        input:
        rect_index: (int) Index for the target rectangle
        img_rects: (list) List of rectangles to be considered

        output: (list) The bounding box for the closest rectangle to the
        left of the target
        '''
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


    def _is_subscript(self, rect_index, img_rects):
        '''
        Checks whether the symbol is a subscript or not
        NOT IMPLEMENTED YET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        input:
        rect_index: (int) The index to check if is a subscript
        img_rects: (list) List of rectangles to be considered

        output: (boolean) True if rect_index is a subscript
        '''
        rect = self.rects[rect_index]
        left = self._get_left(rect_img_rects)
        mask = map(lambda r: all(left == r), img_rects)
        left_index = np.where(mask)[0][0]
        left_label = self.label_dict[left_index]
        if left_label == '-':
            return False


if __name__ == '__main__':
    img = cv2.imread('test4.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = find_symbols(img)

    # mdl = joblib.load('models/cnn.pkl')
    symbols = generate_latex(mdl, img, rects)
