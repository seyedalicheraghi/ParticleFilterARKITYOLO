import numpy as np
import cv2
import glob
from time import time
from numba import jit
from math import sin, cos, sqrt, pi, radians, acos, tan, floor
import math
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import coremltools
from PIL import Image, ImageDraw, ImageFont

R = 11
num_points = 10000
xx = (0, 0)
theta = np.random.uniform(0, 2 * np.pi, num_points)
radius = np.random.uniform(0, R, num_points) ** 0.5
XY = np.vstack((radius * np.cos(theta), radius * np.sin(theta))).T + xx
# print(XY)
# # visualize the points:
# plt.scatter(XY[:, 0], XY[:, 1], s=1)
# plt.show()

img = cv2.imread('../maps/walls_4.bmp', 0)
img1 = cv2.bitwise_not(cv2.imread('../maps/walls_4.bmp'))
height, width = img.shape
mask = np.zeros((height, width), np.uint8)
circle_img = cv2.circle(mask, (126, 94), 50, (255, 255, 255), thickness=-1)
masked_data = cv2.bitwise_not(cv2.bitwise_and(img1, img1, mask=circle_img))
cv2.imshow('1', img)
cv2.imshow('2', masked_data)
cv2.waitKey(0)
