# Import standard libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import draw_3d, detector, superimpose as si
import json

if __name__ == '__main__':
    cols, rows = 2304, 1536
    img = cv2.imread("./sample.png")
    mask = cv2.imread("./PerspectiveMask.png")
    i = 0
    h_mat = np.load("./h_mat.npy")
    img_warped = cv2.warpPerspective(img, h_mat, (rows, rows))
    cv2.imwrite("warpOutput.png", img_warped)


def detect_grid(img, contours):
    for contour in contours:
        i = i + 1
        contour_poly_curve = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, closed=True), closed=True)
        if len(contour_poly_curve) == 4 and i == 7:
            h_mat, inv_h = si.get_h_matrices(contour_poly_curve, rows, rows)
            img_warped = cv2.warpPerspective(img, h_mat, (rows, rows))
            img_warped = cv2.transpose(img_warped)
            cv2.drawContours(img, [contour], 0, (0, 0, 225), 1)
            print(h_mat)
