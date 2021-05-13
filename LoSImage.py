import numpy as np
import cv2
from numba import jit
from math import sin, cos


@jit(nopython=True)
def line_draw(startXY, endXY):
    xy = []
    if (endXY[0] - startXY[0]) is not 0:
        slope = (endXY[1] - startXY[1]) / (endXY[0] - startXY[0])
        if slope >= 0:
            # Case in which starting x and y are smaller than end x and y
            if endXY[0] > startXY[0]:
                # In this case we try to estimate to get increment X and find Y or vice versa
                if abs(endXY[0] - startXY[0]) >= abs(endXY[1] - startXY[1]):  # Add X
                    for i in range(startXY[0], endXY[0]):
                        y = startXY[1] + slope * (i - startXY[0])
                        xy.append(tuple((int(round(i)), int(round(y)))))
                        # cv2.circle(canvas, tuple((int(round(i)), int(round(y)))), 1, (128, 255, 255), -1)
                else:
                    for i in range(startXY[1], endXY[1]):
                        x = startXY[0] + (1 / slope) * (i - startXY[1])
                        xy.append(tuple((int(round(x)), int(round(i)))))
                        # cv2.circle(canvas, tuple((int(round(x)), int(round(i)))), 1, (128, 255, 255), -1)
            else:
                if abs(endXY[0] - startXY[0]) >= abs(endXY[1] - startXY[1]):  # Add X
                    for i in range(endXY[0], startXY[0]):
                        y = startXY[1] + slope * (i - startXY[0])
                        xy.append(tuple((int(round(i)), int(round(y)))))
                        # cv2.circle(canvas, tuple((int(round(i)), int(round(y)))), 1, (128, 255, 255), -1)
                else:
                    for i in range(endXY[1], startXY[1]):
                        x = startXY[0] + (1 / slope) * (i - startXY[1])
                        xy.append(tuple((int(round(x)), int(round(i)))))
                        # cv2.circle(canvas, tuple((int(round(x)), int(round(i)))), 1, (128, 255, 255), -1)
        else:
            if endXY[0] < startXY[0]:
                if abs(endXY[0] - startXY[0]) >= abs(endXY[1] - startXY[1]):  # Add X
                    for i in range(endXY[0], startXY[0]):
                        y = startXY[1] + slope * (i - startXY[0])
                        xy.append(tuple((int(round(i)), int(round(y)))))
                        # cv2.circle(canvas, tuple((int(round(i)), int(round(y)))), 1, (128, 255, 255), -1)
                else:
                    for i in range(startXY[1], endXY[1]):
                        x = startXY[0] + (1 / slope) * (i - startXY[1])
                        xy.append(tuple((int(round(x)), int(round(i)))))
                        # cv2.circle(canvas, tuple((int(round(x)), int(round(i)))), 1, (128, 255, 255), -1)
            else:
                if abs(endXY[0] - startXY[0]) >= abs(endXY[1] - startXY[1]):  # Add X
                    for i in range(startXY[0], endXY[0]):
                        y = startXY[1] + slope * (i - startXY[0])
                        xy.append(tuple((int(round(i)), int(round(y)))))
                        # cv2.circle(canvas, tuple((int(round(i)), int(round(y)))), 1, (128, 255, 255), -1)
                else:
                    for i in range(endXY[1], startXY[1]):
                        x = startXY[0] + (1 / slope) * (i - startXY[1])
                        xy.append(tuple((int(round(x)), int(round(i)))))
                        # cv2.circle(canvas, tuple((int(round(x)), int(round(i)))), 1, (128, 255, 255), -1)
    elif endXY[0] - startXY[0] is 0:
        if endXY[1] > startXY[1]:
            for i in range(startXY[1], endXY[1]):
                xy.append(tuple((int(round(startXY[0])), int(round(i)))))
                # cv2.circle(canvas, tuple((int(round(startXY[0])), int(round(i)))), 1, (128, 255, 255), -1)
        else:
            for i in range(endXY[1], startXY[1]):
                xy.append(tuple((int(round(startXY[0])), int(round(i)))))
                # cv2.circle(canvas, tuple((int(round(startXY[0])), int(round(i)))), 1, (128, 255, 255), -1)
    return xy


@jit(nopython=True)
def mapping(imBinary, imgOriginal, EXITS_X_Y, image_matrix, accuracy):
    for theta in np.arange(0, 360, accuracy):
        for dist in range(0, max(imgOriginal.shape)):
            for exits_xy in EXITS_X_Y:
                newCRD = (int(round(exits_xy[0] + dist * cos(theta))), int(round(exits_xy[1] + dist * sin(theta))))
                # Through the following condition I check if points are within image and not on walls
                if WIDTH > newCRD[0] > 0 and HEIGHT > newCRD[1] > 0 and imBinary[newCRD[1], newCRD[0]] != 255:
                    # Check if a point has direct line of sight to a sign or not
                    coordinates = line_draw(newCRD, exits_xy)
                    flag = True
                    for items in coordinates:
                        if imBinary[items[1], items[0]] == 255:
                            flag = False
                    if flag:
                        image_matrix[newCRD[1], newCRD[0]] = 255
    return image_matrix


flr = 2
path_to_map = '../maps/walls_' + str(flr) + '.bmp'
# EXIT_XY = [(321, 50), (395, 86), (394, 158), (293, 159), (279, 177), (71, 177), (73, 52), (100, 48)]  # 4th Floor
# EXIT_XY = [(99, 44), (72, 51), (71, 190), (270, 174), (421, 188), (421, 49)]  # 3th Floor user_selection_3.bmp
EXIT_XY = [(335, 57), (415, 51), (415, 178), (362, 175), (314, 164), (100, 47)]  # 2th Floor user_selection_2.bmp
# Read Image
image_Original = cv2.imread(path_to_map, 1)
image_Binary = image_Original.copy()[:, :, 0]
image_Binary[image_Binary > 64] = 255
image_Binary[image_Binary <= 64] = 0
HEIGHT, WIDTH, dim = image_Original.shape

exit_matrix = np.zeros(image_Original.shape[0:2], dtype=np.uint8)
accuracy = 10
mapping(image_Binary, image_Original, EXIT_XY, exit_matrix, accuracy)
accuracy = 0.01
mapped = mapping(image_Binary, image_Original, EXIT_XY, exit_matrix, accuracy)
cv2.imwrite('./exits_' + str(flr) + '.bmp', mapped)
