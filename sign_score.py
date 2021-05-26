from numba import jit
import cv2
from math import sin, cos
import numpy as np
import random


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


path_to_map = '../maps/walls_4.bmp'
# Read Image
image_original = cv2.imread(path_to_map, 1)
imBinary = image_original[:, :, 0].copy()
imBinary[imBinary > 64] = 255
imBinary[imBinary <= 64] = 0
Exit_X_Y = [(94, 47), (68, 178), (278, 178), (292, 157), (392, 157), (392, 85)]
for ex in Exit_X_Y:
    cv2.circle(image_original, tuple(ex), 1, (255, 0, 255), 2)
dis = 5
scale = 11.23
r = scale * dis
HEIGHT, WIDTH, dim = image_original.shape
xy = []
for theta in range(0, 360):
    for exits_xy in Exit_X_Y:
        newCRD = (int(round(exits_xy[0] + r * cos(theta))), int(round(exits_xy[1] + r * sin(theta))))
        # Through the following condition I check if points are within image and not on walls
        if WIDTH > newCRD[0] > 0 and HEIGHT > newCRD[1] > 0 and imBinary[newCRD[1], newCRD[0]] != 255:
            # Check if a point has direct line of sight to a sign or not
            coordinates = line_draw(newCRD, exits_xy)
            flag = True
            for items in coordinates:
                if imBinary[items[1], items[0]] == 255:
                    flag = False
            if flag:
                xy.append(newCRD)
for items in xy:
    image_original[items[1], items[0]] = (128, 255, 255)

smsize = 100
xy = np.array(xy)
index = np.random.choice(len(xy), smsize)
additionalError = np.vstack((np.random.randint(-int(round(scale)), int(round(scale)), smsize),
                             np.random.choice(int(round(scale)), smsize))).T
newArray = xy[index] + additionalError
newArray = newArray[newArray[:, 1] < HEIGHT]
newArray = newArray[newArray[:, 1] < 0]
newArray = newArray[newArray[:, 0] < WIDTH]
newArray = newArray[newArray[:, 0] > 0]
print(newArray.shape)
print(imBinary[newArray[:, 1], newArray[:, 0]])
exit(0)
newArray = imBinary[newArray] != 255
for it in newArray:
    cv2.circle(image_original, tuple((int(round(it[0])), int(round(it[1])))), 1, (128, 255, 255), -1)
cv2.imshow("WINDOW_NAME_PARTICLES", image_original)
cv2.waitKey(0)
