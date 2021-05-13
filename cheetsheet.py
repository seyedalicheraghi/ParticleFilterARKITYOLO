import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2

flr = 3
PATH = "../LoggedData/trial" + str(flr) + "/"
IMAGE_EXTENSION = '.jpg'


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


# Read the TXT ARKIT data
ARKIT_LOGGED = [x for x in open(glob.glob(PATH + "*.txt")[0], "r")]
# Read Images and sort them
listOfImages = [int(file.split('/')[len(file.split('/')) - 1].replace(IMAGE_EXTENSION, ''))
                for file in glob.glob(PATH + "*" + IMAGE_EXTENSION)]

listOfImages.sort()
pointX = []
pointY = []
imgPcv2 = np.zeros((30, 50, 3), dtype=np.uint8)
imgP = np.zeros((30, 50, 3), dtype=np.uint8)
# Start to read each image and its ARKIT Logged data
for image in listOfImages:
    X_ARKIT = XYZ_pitch_yaw_roll(str(image), ARKIT_LOGGED)[0]
    Y_ARKIT = XYZ_pitch_yaw_roll(str(image), ARKIT_LOGGED)[1]
    Z_ARKIT = XYZ_pitch_yaw_roll(str(image), ARKIT_LOGGED)[2]
    PITCH = XYZ_pitch_yaw_roll(str(image), ARKIT_LOGGED)[3]
    YAW = XYZ_pitch_yaw_roll(str(image), ARKIT_LOGGED)[4]
    ROLL = XYZ_pitch_yaw_roll(str(image), ARKIT_LOGGED)[5]
    cv2.circle(imgPcv2, center=(round(-Z_ARKIT)+10, round(X_ARKIT)+10), radius=1, color=(0, 0, 255), thickness=-1)
    imgP[int(round(X_ARKIT))+10, int(round(-Z_ARKIT))+10] = (0, 0, 255)
#     pointX.append(-Z_ARKIT)
#     pointY.append(X_ARKIT)
# plt.scatter(pointX, pointY)
# plt.show()
cv2.imwrite('./cv2.bmp', imgPcv2)
cv2.imwrite('./python.bmp', imgP)
