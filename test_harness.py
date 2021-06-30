from ParticleFilter import main, mouse_click
from map_gui import get_location_in_image
import numpy as np
import cv2
import glob
import os.path
import pickle
from numba import jit
from math import sin, cos, sqrt, pi, radians, acos, tan, floor
import matplotlib.pyplot as plt
import coremltools

img = []
flr = 4
PATH = "../LoggedData/trial" + str(flr) + "/"
IMAGE_EXTENSION = '.jpg'
path_to_map = '../maps/walls_' + str(flr) + '.bmp'
model_name = '8NMMY2NC15k.mlmodel'
class_names = ['Safety', 'Exit', 'FaceMask', 'James', 'Caution', 'RedFire', 'Restroom', 'SixFt']
class_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (192, 192, 192), (192, 0, 0), (0, 192, 0), (0, 0, 192)]
# I refers to Inches and M refers to meters
# FireHose: 38 I -> 0.9652 M
# Sanitizer: 9 I -> 0.2286 M
# Exit Height:      4-1/2 I -> 0.19685 M    ----  Exit Width:
# FireAlarm Height: 4-1/2 I -> 0.1143 M    ----  FireAlarm Width: 3 I -> 0.0762 M
# Restroom Height: 11-3/4 I -> 0.29845 M   ----  Restroom Width: 18 I -> 0.4572 M
# FaceMask Height: 11 I -> 0.2794 M        ----  FaceMask Width: 8.5 I ->  0.2159M
# Six Feet Height: 11 I -> 0.2794 M        ----  FaceMask Width: 8.5 I ->  0.2159M
# James Height:    11 I -> 0.2794 M        ----  James Width:    8.5 I ->  0.2159M
Heights = [0.2794, 0.19685, 0.2794, 0.2794, 0.2794, 0.1143, 0.29845, 0.2794]  # Heights of signs in meters
if flr is 4:
    EXITS_X_Y = [(321, 50), (395, 86), (394, 158), (293, 159), (279, 177), (71, 177), (73, 52), (100, 48)]  # 4th Flr
if flr is 3:
    EXITS_X_Y = [(99, 44), (72, 51), (71, 190), (270, 174), (421, 188), (421, 49)]  # 3th Flr
if flr is 2:
    EXITS_X_Y = [(335, 57), (415, 51), (415, 178), (362, 175), (314, 164), (100, 47)]  # 2th Flr
EXITS_MAP = cv2.bitwise_not(cv2.imread('../LOS_Maps/exits_' + str(flr) + '.bmp'))
FocalLengths = [1602]
model_input_size = 416
NumberOfParticles = 100000
# Number of particles
# Read Image
imgOriginal = cv2.imread(path_to_map, 1)

# Sample distance define how far from a selected starting point we need to do sampling
sample_distance = 2
Scale = 12
randseed = 542014
loadedModel = coremltools.models.MLModel('../DeepLearning/TrainedModels/' + model_name)
_, imgOriginal = cv2.threshold(imgOriginal, 64, 255, cv2.THRESH_BINARY)
image_binary = imgOriginal.copy()[:, :, 0]
Im_HEIGHT, Im_WIDTH, dim = imgOriginal.shape
# make imgOriginal a single-channel grayscale image if it's originally an RGB
imageP = imgOriginal.copy()[:, :, 0] if len(imgOriginal.shape) == 3 else imgOriginal.copy()
# Get VIO data filename and create corresponding ground truth filename
VIO_file = glob.glob(PATH + "*.txt")[0]
GT_file = VIO_file.replace(".txt", ".gt")
# Read the TXT ARKIT data
ARKITLOG = [x.strip() for x in open(VIO_file, "r")]
# Read Images and sort them
ListOfCameraLogImages = [int(file.split('/')[len(file.split('/')) - 1].replace(IMAGE_EXTENSION, ''))
                         for file in glob.glob(PATH + "*" + IMAGE_EXTENSION)]
ListOfCameraLogImages.sort()
# Check for existing ground truth
if os.path.isfile(GT_file):
    # Read in existing ground truth
    print(f"Reading ground truth data from file {GT_file}.\n")
    with open(GT_file, 'rb') as fp:
        gt = pickle.load(fp)
    user_initial_location = gt[0]
    print(f"Initial location is at: ({user_initial_location[0]}, {user_initial_location[1]}).\n")
        # gt = [[x.strip().split(",")] for x in open(GT_file, "r")]
else:  # otherwise create the list
    # Get start point
    user_initial_location = get_location_in_image(imgOriginal.copy())
    print(f"Initial location is at: ({user_initial_location[0]}, {user_initial_location[1]}).\n")
    gt = main(imgOriginal, imageP, 10000, PATH, ListOfCameraLogImages, ARKITLOG, Scale, Im_HEIGHT, Im_WIDTH,
              loadedModel, class_names, class_colors, model_input_size, Heights, FocalLengths, EXITS_X_Y, EXITS_MAP,
              (user_initial_location is not None), user_initial_location, sample_distance, 0, randseed)
    # and save it
    print(f"Writing ground truth data to file {GT_file}.\n")
    with open(GT_file, 'wb') as fp:
        pickle.dump(gt, fp)

# test harness loop
ntrials = 10
yawflag = None
yolo_flag = True
trajectoryDistThresh = 1  # meters away from ground truth peaks must be to count correct
convergetime = []
np.random.seed(randseed)
randseeds = (np.random.randint(low=1, high=1000000, size=ntrials))
for trial in range(ntrials):
    peakpath = main(imgOriginal, imageP, NumberOfParticles, PATH, ListOfCameraLogImages, ARKITLOG, Scale, Im_HEIGHT, Im_WIDTH,
                    loadedModel, class_names, class_colors, model_input_size, Heights, FocalLengths, EXITS_X_Y,
                    EXITS_MAP, False, user_initial_location, sample_distance, yawflag, randseeds[trial], yolo_flag)
    # Going from the end, find the last step in which the peak was not correct
    for step in reversed(range(len(peakpath))):
        dist = sqrt((gt[step][0]-peakpath[step][0])**2 + (gt[step][1]-peakpath[step][1])**2)
        if dist > Scale*trajectoryDistThresh:
            convergetime.append(step)
            break
mean_conv = sum(convergetime)/len(convergetime)
print(f"Sign detection is set to {yolo_flag}.\n The total number of times points was {len(gt)}.\n")
print(f"Over {ntrials} trials with {NumberOfParticles} particles, the average time for peak to converge correctly is {mean_conv}.\n")
print(f"It converged {len(convergetime)} times out of {ntrials} trials.\n")
print(f"The individual times were:\n")
print(convergetime)
