"""
This script gets an image as an input and then print out areas selected by the user.
These selections later on will be used to define locations of signs within a floor plan used by the particle filter.
"""
import numpy as np
import cv2
import glob
from numba import jit
from math import sin, cos, sqrt, pi, radians, acos, tan, floor
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import coremltools
from PIL import Image, ImageDraw, ImageFont

img = []


# define the events for the mouse_click.
def mouse_click(event, x, y, flags, param):
    # to check if left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # display that left button was clicked.
        cv2.circle(img, (x, y), radius=2, color=(255, 255, 0), thickness=3)
        cv2.imshow('image', img)
        print((x, y))


if __name__ == "__main__":
    path_to_map = '../maps/walls_4.bmp'
    img = cv2.imread(path_to_map)
    # show image
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', mouse_click)
    cv2.waitKey(0)
