import numpy as np
import cv2
import glob
from math import radians, floor
import matplotlib.pyplot as plt


def XYZ_pitch_yaw_roll(imagesName, ARKIT):
    # read the CoreML txt file
    for items in range(0, len(ARKIT)):
        if ARKIT[items].strip() == imagesName.strip():
            xyz = ARKIT[items + 4].split(',')
            sp = ARKIT[items + 5].split('<Float>')
            sp1 = sp[1].split(',')
            sp2 = sp1[0].split('(')
            sp3 = sp1[2].split(')')
            pitch = float(sp2[1])
            yaw = float(sp1[1].strip())
            roll = float(sp3[0].strip())
            return float(xyz[0].strip()), float(xyz[1].strip()), float(xyz[2].strip()), pitch, yaw, roll


flr = 3
Scale = 11.7
initial_location = (74, 45)  # Assigned starting location of the user
particles = np.array([initial_location[0], initial_location[1], 0])  # Create a particle located on starting point
imgOriginal = cv2.imread('./maps/walls_' + str(flr) + '.bmp', 1)  # Read the Map
previous_RAW_YAW = 0
previous_X = 0
previous_Z = 0
# Read the TXT ARKIT data
ARKIT_LOGGED = [x for x in open(glob.glob("./LoggedData/flr" + str(flr) + ".txt")[0], "r")]
# Show where starting location
# cv2.circle(imgOriginal, center=initial_location, color=(0, 255, 0), thickness=-1, radius=3)
img_trj = imgOriginal.copy()  # Image Trajectory
cv2.circle(img_trj, tuple((int(round(particles[0])), int(round(particles[1])))), 2, (128, 255, 255), 2)
# Start to read each image and its ARKIT Logged data
for CameraLogImage in range(1, 20000, 1):
    ARKIT_DATA = XYZ_pitch_yaw_roll(str(CameraLogImage), ARKIT_LOGGED)
    if ARKIT_DATA is None:
        exit(0)
    X_ARKIT = ARKIT_DATA[0] * Scale
    Z_ARKIT = -ARKIT_DATA[2] * Scale
    image_trajectory_y = int(round(X_ARKIT)) + initial_location[1]
    image_trajectory_x = int(round(Z_ARKIT)) + initial_location[0]
    cv2.circle(img_trj, tuple((image_trajectory_x, image_trajectory_y)), 1, (128, 255, 255), 1)
    RAW_YAW = ARKIT_DATA[4]
    img_prt = imgOriginal.copy()  # To show particles
    Rot2D_theta = particles[2] - np.pi / 2 - previous_RAW_YAW
    DeltaX = X_ARKIT - previous_X
    DeltaZ = Z_ARKIT - previous_Z

    DU = DeltaX * np.cos(Rot2D_theta) + DeltaZ * np.sin(Rot2D_theta)
    DV = DeltaX * np.sin(Rot2D_theta) - DeltaZ * np.cos(Rot2D_theta)
    particles[0] = particles[0] + DU
    particles[1] = particles[1] + DV
    particles[2] = particles[2] + (RAW_YAW - previous_RAW_YAW)
    # -----------------------Updating Parameters--------------------------
    cv2.circle(img_prt, tuple((int(round(particles[0])), int(round(particles[1])))), 2, (128, 255, 255), 2)

    previous_RAW_YAW = RAW_YAW
    previous_X = X_ARKIT
    previous_Z = Z_ARKIT
    output = np.hstack((img_trj, img_prt))
    cv2.imshow("Particles", output)
    # -----------------------Waiting a little bit to show all windows-----
    if cv2.waitKey(1) & 0xFF == 27:
        break
    plt.pause(0.000001)
