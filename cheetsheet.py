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


def main(image_original, number_of_particles, list_of_cameraLog_images, scale, user_starting_point):
    sampling_distance = 1
    # Create particles located on empty spaces
    particles_data = generate_uniform_particles_data(number_of_particles, image_original, user_starting_point,
                                                     floor(sampling_distance * scale))
    previous_RAW_YAW = 0
    previous_X = 0
    previous_Z = 0
    image_trajectory = image_original.copy()

    # Start to read each image and its ARKIT Logged data
    for CameraLogImage in list_of_cameraLog_images:
        ARKIT_DATA = XYZ_pitch_yaw_roll(str(CameraLogImage), ARKIT_LOGGED)
        X_ARKIT = ARKIT_DATA[0] * scale
        Z_ARKIT = -ARKIT_DATA[2] * scale

        image_trajectory_y = int(round(X_ARKIT)) + user_starting_point[1]
        image_trajectory_x = int(round(Z_ARKIT)) + user_starting_point[0]
        cv2.circle(image_trajectory, tuple((image_trajectory_x, image_trajectory_y)), 2, (128, 255, 255), 2)
        RAW_YAW = ARKIT_DATA[4]
        image_particles = image_original.copy()
        Rot2D_theta = particles_data[:, 2] - np.pi/2 - previous_RAW_YAW
        DeltaX = X_ARKIT - previous_X
        DeltaZ = Z_ARKIT - previous_Z
        # DeltaZ = previous_Z - Z_ARKIT

        DU = DeltaX * np.cos(Rot2D_theta) + DeltaZ * np.sin(Rot2D_theta)
        DV = DeltaX * np.sin(Rot2D_theta) - DeltaZ * np.cos(Rot2D_theta)
        particles_data[:, 0] = particles_data[:, 0] + DU
        particles_data[:, 1] = particles_data[:, 1] + DV
        particles_data[:, 2] = particles_data[:, 2] + (RAW_YAW - previous_RAW_YAW)
        # particles_data[:, 2] = particles_data[:, 2] + (previousYAW - YAW)
        # -----------------------Updating Parameters--------------------------

        for particle in particles_data:
            cv2.circle(image_particles, tuple((int(round(particle[0])), int(round(particle[1])))), 3, (128, 255, 255), 5)

        previous_RAW_YAW = RAW_YAW
        previous_X = X_ARKIT
        previous_Z = Z_ARKIT
        cv2.imshow('Trajectory', image_trajectory)
        cv2.imshow("Particles", image_particles)
        # -----------------------Waiting a little bit to show all windows-----
        if cv2.waitKey(1) & 0xFF == 27:
            break
        plt.pause(0.000001)


img = []
while_flag = True
# Change the flag to change modes between known or unknown starting location
initiated_flag = True
# This flag is responsible to pause and continue the process
control_flag = False
# This section tries to assign starting location of the user
user_initial_location = (94, 47)


# define the events for the mouse_click.
def mouse_click(event, x, y, flags, param):
    global user_initial_location
    global while_flag
    global initiated_flag
    global control_flag

    if not control_flag:
        # to check if left mouse button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # display that left button was clicked.
            cv2.circle(img, (x, y), radius=2, color=(255, 255, 0), thickness=3)
            cv2.imshow('image', img)
            user_initial_location = (x, y)
            while_flag = False
            initiated_flag = True
            control_flag = True
        if event == cv2.EVENT_RBUTTONDOWN:
            while_flag = False
            initiated_flag = False
            control_flag = True
    else:
        # to check if left mouse button was clicked for pausing
        if event == cv2.EVENT_LBUTTONDOWN:
            if while_flag:
                while_flag = False
            else:
                while_flag = True
        # Check if right mouse button was clicked for ending the program
        if event == cv2.EVENT_RBUTTONDOWN:
            exit(0)


if __name__ == "__main__":
    flr = 2
    PATH = "../LoggedData/trial" + str(flr) + "/"
    IMAGE_EXTENSION = '.jpg'
    path_to_map = '../maps/walls_' + str(flr) + '.bmp'
    # Number of particles
    NumberOfParticles = 1
    # Read Image
    imgOriginal = cv2.imread(path_to_map, 1)
    img = imgOriginal.copy()
    # show image
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', mouse_click)
    while while_flag:
        cv2.waitKey(2)
    cv2.destroyWindow('image')
    Scale = 11.7
    _, imgOriginal = cv2.threshold(imgOriginal, 64, 255, cv2.THRESH_BINARY)
    Im_HEIGHT, Im_WIDTH, dim = imgOriginal.shape
    # Read the TXT ARKIT data
    ARKIT_LOGGED = [x for x in open(glob.glob(PATH + "*.txt")[0], "r")]
    # Read Images and sort them
    ListOfCameraLogImages = [int(file.split('/')[len(file.split('/')) - 1].replace(IMAGE_EXTENSION, ''))
                             for file in glob.glob(PATH + "*" + IMAGE_EXTENSION)]
    ListOfCameraLogImages.sort()
    cv2.circle(imgOriginal, center=user_initial_location, color=(0, 255, 0), thickness=-1, radius=3)
    main(imgOriginal, NumberOfParticles, ListOfCameraLogImages, Scale, user_initial_location)
