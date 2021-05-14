#analyze Gio's calculations
#originally written in Dec 2017, updated to Python 3 in Aug 2018
#v2 rotation from Gio on 8/22/18
#updated 2/17/21 to reflect fact that hold() is no longer used in matplotlib
#updated 5/14/21 to correspond to equations in pdf doc

from math import sin, cos, pi
import numpy as np
from pylab import close, colorbar, figure, gray, hist, imshow, ion, plot, savefig, show, title
import scipy
import imageio
import csv

close('all')
ion()

#load Gio's data. 
#.csv format: the columns are: VIO_Theta, VIO_X, VIO_Z, Marker_Theta, MARKER_X, MARKER_Z
#fname = 'data.csv'
fname = 'data_log_123.csv'

data=[]
with open(fname, 'r') as f: #original version had 'rb' as 2nd argument, had to change to 'r' to work in Python 3
	reader = csv.reader(f)
	for row in reader:
		data.append(row)
f.close()

#extract data
n = len(data)
VIO_list, marker_list, marker_detected = [], [], []

for k, item in enumerate(data):
	VIO_list.append((eval(item[0]),eval(item[1]),eval(item[2]))) #VIO_Theta, VIO_X, VIO_Z
	if len(item)>3: #marker data is also present
		marker_list.append((eval(item[3]),eval(item[4]),eval(item[5]))) #Marker_Theta, MARKER_X, MARKER_Z
		marker_detected.append(True)
	else:
		marker_list.append([])
		marker_detected.append(False)

#convert to arrays:
VIOs = np.array(VIO_list)
markers = VIOs*0
for k, item in enumerate(data):
	if len(item)>3: #marker data is also present
		markers[k] = np.array(marker_list[k])

#floor plan (map)
map_image = imageio.imread('Walls.png')
h,w,_ = np.shape(map_image)
scale = 33.56/1.4859  #pixels per meter

#[(markers[k][0] - VIOs[k][0])*180/pi for k in range(n) if marker_available_list[k]]
#the above line shows that the VIO yaw is pretty consistent with marker yaw -- usually only a few to 7 degrees discrepancy

#plot locations estimated from marker detections:
Xs = np.array([markers[k][1] for k in range(n) if marker_detected[k]])
Zs = np.array([markers[k][2] for k in range(n) if marker_detected[k]])

figure();title('Just locations from marker detections')
#hold(True)
imshow(map_image)
plot(Xs*scale,(h-1.)-Zs*scale,'ro-')
#hold(False)

#now use VIO when marker isn't available
xs,zs,yaws = np.zeros((n,),float),np.zeros((n,),float),np.zeros((n,),float)

#set to marker values when these are available, otherwise use estimate from last observed marker and propagate that to current frame using VIO:
k_star = 0 #the last frame index when marker was visible
for k in range(n):
	if marker_detected[k]:
		yaws[k],xs[k],zs[k] = markers[k][0],markers[k][1],markers[k][2]
		k_star = k + 0 #the last frame index when marker was visible
	else:
		#yaws[k] = markers[k][0] + (VIOs[k][0] - VIOs[k_star][0])
		delta_ang = markers[k_star][0] - VIOs[k_star][0] # would also have -pi/2 as in pdf, but for a while Gio used a different yaw convention: yaw_GIO = yaw - pi/2
		delta_X = VIOs[k][1]-VIOs[k_star][1]
		delta_Z = VIOs[k][2]-VIOs[k_star][2]
		
		xs[k] = markers[k_star][1] + cos(delta_ang)*delta_X + sin(delta_ang)*delta_Z
		zs[k] = markers[k_star][2] + sin(delta_ang)*delta_X - cos(delta_ang)*delta_Z
		
figure();title('Integrated: yellow triangle indicates marker was detected.')
#hold(True)
imshow(map_image)
plot(np.array(xs)*scale,(h-1.)-np.array(zs)*scale,'ro-',markersize = 10, fillstyle = 'none')
for k in range(n):
    if marker_detected[k]: #add special symbol, a yellow triangle
        plot(xs[k]*scale,(h-1.)-zs[k]*scale,'y^',markersize = 5)
        print('marker detected',k)
#hold(False)
