import glob
import cv2


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
scale = 11.7
PATH = "../LoggedData/trial" + str(flr) + "/"
IMAGE_EXTENSION = '.jpg'
path_to_map = '../maps/walls_' + str(flr) + '.bmp'
image_trajectory = cv2.imread(path_to_map, 1)
ListOfCameraLogImages = [int(file.split('/')[len(file.split('/')) - 1].replace(IMAGE_EXTENSION, ''))
                         for file in glob.glob(PATH + "*" + IMAGE_EXTENSION)]
user_starting_point_flr = (79, 43)
ListOfCameraLogImages.sort()
ARKIT_LOGGED = [x for x in open(glob.glob(PATH + "*.txt")[0], "r")]
for CameraLogImage in ListOfCameraLogImages:
    ARKIT_DATA = XYZ_pitch_yaw_roll(str(CameraLogImage), ARKIT_LOGGED)
    X_ARKIT = ARKIT_DATA[0] * scale
    Z_ARKIT = -ARKIT_DATA[2] * scale
    x = int(round(X_ARKIT)) + user_starting_point_flr[1]
    y = int(round(Z_ARKIT)) + user_starting_point_flr[0]
    image_trajectory[x, y] = (255, 255, 0)
    cv2.imshow('image', image_trajectory)
    cv2.waitKey(50)
