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


def generate_uniform_particles_data(number_of_particles, image_original, starting_point, R):
    h, w, d = image_original.shape
    mask = np.zeros((h, w), np.uint8)
    circle_img = cv2.circle(mask, starting_point, R, (255, 255, 255), thickness=-1)
    img1 = cv2.bitwise_not(image_original.copy())
    masked_data = cv2.bitwise_not(cv2.bitwise_and(img1, img1, mask=circle_img))
    WalkableAreas = np.where(masked_data[:, :, 0] == 0)
    WalkableAreas = np.transpose((WalkableAreas[1], WalkableAreas[0])).astype(float)
    rnd_indexes = np.random.choice(WalkableAreas.shape[0], number_of_particles, replace=True)
    rnd_particles = WalkableAreas[rnd_indexes]
    rnd_yaw = np.reshape(np.array([radians(hd) for hd in np.random.uniform(-180, 180, size=number_of_particles)]),
                         (number_of_particles, 1)) * 0

    probability_distribution = np.ones((number_of_particles, 1), dtype=np.float64)
    probability_distribution /= np.sum(probability_distribution)
    output_particles = np.hstack((rnd_particles, rnd_yaw, probability_distribution))
    return output_particles


flr = 3
Scale = 11.7
NumberOfParticles = 5  # Number of particles
initial_location = (74, 50)  # Assigned starting location of the user
imgOriginal = cv2.imread('./maps/walls_' + str(flr) + '.bmp', 1)  # Read the Map
previous_RAW_YAW = 0
previous_X = 0
previous_Z = 0
# Read the TXT ARKIT data
ARKIT_LOGGED = [x for x in open(glob.glob("./LoggedData/flr" + str(flr) + ".txt")[0], "r")]
# Show where on the map the user clicked
cv2.circle(imgOriginal, center=initial_location, color=(0, 255, 0), thickness=-1, radius=3)
# Create particles located on empty spaces
particles = generate_uniform_particles_data(NumberOfParticles, imgOriginal, initial_location, floor(Scale))
img_trj = imgOriginal.copy()  # Image Trajectory

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
    Rot2D_theta = particles[:, 2] - np.pi / 2 - previous_RAW_YAW
    DeltaX = X_ARKIT - previous_X
    DeltaZ = Z_ARKIT - previous_Z

    DU = DeltaX * np.cos(Rot2D_theta) + DeltaZ * np.sin(Rot2D_theta)
    DV = DeltaX * np.sin(Rot2D_theta) - DeltaZ * np.cos(Rot2D_theta)
    particles[:, 0] = particles[:, 0] + DU
    particles[:, 1] = particles[:, 1] + DV
    particles[:, 2] = particles[:, 2] + (RAW_YAW - previous_RAW_YAW)
    # -----------------------Updating Parameters--------------------------
    for particle in particles:
        cv2.circle(img_prt, tuple((int(round(particle[0])), int(round(particle[1])))), 3, (128, 255, 255), 5)

    previous_RAW_YAW = RAW_YAW
    previous_X = X_ARKIT
    previous_Z = Z_ARKIT
    output = np.hstack((img_trj, img_prt))
    cv2.imshow("Particles", output)
    # -----------------------Waiting a little bit to show all windows-----
    if cv2.waitKey(1) & 0xFF == 27:
        break
    plt.pause(0.000001)
