import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

W = 400

def my_ellipse(img, angle):
    thickness = 2
    line_type = 8

    cv.ellipse(img, (W // 2, W // 2 ), (W // 4, W // 16), angle, 0, 360, (255, 0, 0), thickness, line_type)


def my_filled_circle(img, center):
    thickness = -1
    line_type = 8

    cv.circle(img, center, W // 32, (0, 0, 255), thickness, line_type)


def my_polygon(img):
    line_type = 8

    ppt = np.array([[W / 4, 7 * W / 8], [3 * W / 4, 7 * W / 8],
                    [3 * W / 4, 13 * W / 16], [11 * W / 16, 13 * W / 16],
                    [19 * W / 32, 3 * W / 8], [3 * W / 4, 3 * W / 8],
                    [3 * W / 4, W / 8], [26 * W / 40, W / 8],
                    [26 * W / 40, W / 4], [22 * W / 40, W / 4],
                    [22 * W / 40, W / 8], [18 * W / 40, W / 8],
                    [18 * W / 40, W / 4], [14 * W / 40, W / 4],
                    [14 * W / 40, W / 8], [W / 4, W / 8],
                    [W / 4, 3 * W / 8], [13 * W / 32, 3 * W / 8],
                    [5 * W / 16, 13 * W / 16], [W / 4, 13 * W / 16]], np.int32)
    ppt = ppt.reshape((-1, 1, 2))
    cv.fillPoly(img, [ppt], (255, 255, 255), line_type)


def my_line(img, start, end):
    thickness = 2
    line_type = 8

    cv.line(img, start, end, (0, 0, 0), thickness, line_type)


atom_window = "Drawing 1: Atom"


size = W, W, 3
atom_image = np.zeros(size, dtype=np.uint8)

my_ellipse(atom_image, 90)
my_ellipse(atom_image, 0)
my_ellipse(atom_image, 45)
my_ellipse(atom_image, -45)

my_filled_circle(atom_image, (W // 2, W // 2))


# plt.imshow(atom_image)
# plt.show()

cv.imshow(atom_window, atom_image)
cv.moveWindow(atom_window, 0, 200)
cv.waitKey(1000)
cv.destroyAllWindows()