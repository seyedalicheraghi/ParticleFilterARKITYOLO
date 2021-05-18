# plot VIO trajectory using known initial location and yaw
# and superimpose on the map

from math import sin, cos, pi
import numpy as np
from pylab import close, figure, imshow, ion, plot
from matplotlib.pyplot import imread, waitforbuttonpress

close('all')
ion()


# ARKit VIO data parser
def loadVIO(filename):
    # Load the VIO txt file
    ARK = [x.strip() for x in open(filename, "r")]
    X = []
    Y = []
    Z = []
    PITCH = []
    YAW = []
    ROLL = []
    STATUS = []
    for idx in range(0, len(ARK)):
        if ARK[idx].isdigit():
            [x, y, z] = [float(x) for x in ARK[idx + 4].split(',')]
            [pitch, yaw, roll] = [float(x) for x in ARK[idx + 5].split('(')[1].strip(')').split(',')]
            normalStatus = ARK[idx + 2] == 'normal'
            X.append(x)
            Y.append(y)
            Z.append(z)
            PITCH.append(pitch)
            YAW.append(yaw)
            ROLL.append(roll)
            STATUS.append(normalStatus)
    return X, Y, Z, PITCH, YAW, ROLL, STATUS


Xs, Ys, Zs, PITCHs, RAW_YAWs, ROLLs, STATUSs = loadVIO('VIO_1620536957.533128.txt')

map_image = imread('./maps/walls_3.bmp')
h, w, _ = np.shape(map_image)
scale = 11.7  # from Ali, I assume pixels/meter

yaw_0 = 0.  # initial yaw
initial_location_pix = (74, 45)  # (col, row) from Ali: assigned starting location of the user for floor 3
initial_location = (initial_location_pix[0] / scale, (h - 1 - initial_location_pix[1]) / scale)

n = len(Xs)
delta_ang = yaw_0 - pi / 2

# use eq. 7, putting everything into meters and switching coordinate names (u,v) to (x,y)
xs, ys = [], []
for k in range(n):
    delta_X = Xs[k] - Xs[0]
    delta_Z = Zs[k] - Zs[0]

    x = initial_location[0] + cos(delta_ang) * delta_X + sin(delta_ang) * delta_Z
    y = initial_location[1] + sin(delta_ang) * delta_X - cos(delta_ang) * delta_Z

    xs.append(x)
    ys.append(y)

figure()
imshow(map_image)
for k in range(n):
    plot(xs[k] * scale, (h - 1.) - ys[k] * scale, 'yo', markersize=5)
waitforbuttonpress(timeout=- 1)
