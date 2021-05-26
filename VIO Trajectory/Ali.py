import numpy as np
import cv2
import glob
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


flr = 4
Scale = 11.7
# initial_location_pix = (74, 45)  # Assigned starting location of the user
initial_location_pix = (224, 48)  # Assigned starting location of the user
imgOriginal = cv2.imread('./maps/walls_' + str(flr) + '.bmp', 1)  # Read the Map
h, w, _ = np.shape(imgOriginal)
initial_location = (initial_location_pix[0] / Scale, (h - 1 - initial_location_pix[1]) / Scale)
particles = np.array([initial_location[0], initial_location[1], 3*(np.pi / 2)])
# particles = np.array([initial_location[0], initial_location[1], 0])
particles_data = particles.copy()
previous_RAW_YAW = 0
previous_X = 0
previous_Y = 0
# Read the TXT ARKIT data
ARKIT_LOGGED = [x for x in open(glob.glob("./LoggedData/flr" + str(flr) + ".txt")[0], "r")]
# Show where starting location
img_trj = imgOriginal.copy()  # Image Trajectory
cv2.circle(img_trj, tuple((initial_location_pix[0], initial_location_pix[1])), 2, (128, 0, 255), 2)
# Start to read each image and its ARKIT Logged data
xs, ys = [], []
for CameraLogImage in range(1, 20000, 1):
    ARKIT_DATA = XYZ_pitch_yaw_roll(str(CameraLogImage), ARKIT_LOGGED)
    if ARKIT_DATA is None:
        exit(0)
    X_ARKIT = ARKIT_DATA[0]
    Y_ARKIT = ARKIT_DATA[2]
    RAW_YAW = ARKIT_DATA[4]
    img_prt = imgOriginal.copy()  # To show particles
    Rot2D_theta = particles_data[2] - np.pi / 2 - previous_RAW_YAW
    DeltaX = X_ARKIT - previous_X
    DeltaY = Y_ARKIT - previous_Y
    U = DeltaX * np.cos(Rot2D_theta) + DeltaY * np.sin(Rot2D_theta)
    V = DeltaX * np.sin(Rot2D_theta) - DeltaY * np.cos(Rot2D_theta)
    # Updating X and Y for each hypothesis
    particles_data[0] = particles_data[0] + U
    particles_data[1] = particles_data[1] + V
    particles_data[2] = particles_data[2] + (RAW_YAW - previous_RAW_YAW)
    previous_RAW_YAW = RAW_YAW
    previous_X = X_ARKIT
    previous_Y = Y_ARKIT
    # -----------------------Updating Parameters--------------------------
    cv2.circle(img_trj, tuple((int(round(particles_data[0] * Scale)), int(round(h - particles_data[1] * Scale)))),
               1, (128, 255, 255), -1)
    cv2.imshow("Particles", img_trj)
    # -----------------------Waiting a little bit to show all windows-----
    if cv2.waitKey(1) & 0xFF == 27:
        break
    plt.pause(0.000001)
