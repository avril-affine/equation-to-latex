import cv2
import matplotlib.pyplot as plt
import numpy as np


def findnumbers(invert, f='test/test3.png'):
    img = cv2.imread(f)
    # print img.shape
    # print type(img)
    # print img.shape
    # print img[0][0]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # plt.imshow(img_gray, 'gray')
    # plt.show()

    ret, img_thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)

    # plt.imshow(img_thresh, 'gray')
    # plt.show()

    contours, hier = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    rects = map(cv2.boundingRect, contours)

    numbers = []
    for rect in rects:
        # print rect
        cv2.rectangle(img, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]),
                      (0, 255, 0), 3)
        plt.imshow(img)
        plt.show()
        bfr = 1.2
        length = int(rect[3] * bfr)
        pt1 = int(rect[1] + rect[3] // 2 - length // 2)
        pt2 = int(rect[0] + rect[2] // 2 - length // 2)

        if invert:
            roi = img_thresh[pt1:pt1+length, pt2:pt2+length]
        # roi = img[pt1:pt1+length, pt2:pt2+length]
        else:
            roi = img_gray[pt1:pt1+length, pt2:pt2+length]
        # plt.imshow(roi, 'gray')
        # plt.show()
        # roi = img_thresh[rect[1]:rect[1] + rect[3],
        #                  rect[0]:rect[0] + rect[2]]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        # roi = cv2.dilate(roi, (3, 3))
        # roi = cv2.GaussianBlur(roi, (5, 5), 0)
        # print roi.shape
        # plt.imshow(roi, 'gray')
        # plt.show()
        numbers.append(roi)

    # plt.imshow(img, 'gray')
    # plt.show()
    return (rects, numbers)


def main():
    findnumbers(False)


if __name__ == '__main__':
    main()
