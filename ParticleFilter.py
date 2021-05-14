import numpy as np
import cv2
import glob
from numba import jit
from math import sin, cos, sqrt, pi, radians, acos, tan, floor
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import coremltools
from PIL import Image, ImageDraw, ImageFont

np.set_printoptions(precision=40)
np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)

# Constant variables to describe particle data array index
Z_ = 0; X_ = 1; YAW_ = 2; SCALE_ = 3; SCORE_ = 4

def rangeZ(h0, gamma, alpha, delta):
    return h0 / (tan(gamma + alpha + delta) - tan(gamma + alpha))


def angle_between_two_pixels(f, u1, v1, u2, v2):  # here (u,v) is (0,0) in center of image
    p1 = np.array([u1, v1, f])
    p2 = np.array([u2, v2, f])
    norm_prod = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    if norm_prod > 1:
        norm_prod = 1

    return acos(norm_prod)


def distance_measuring_new(bottom_row, gamma, h0, f, sz, top_row, col):
    h, w = sz
    # row = 0 is top row in image but (u,v) = (0,0) is center of image, with positive u to right and positive v up
    alpha = angle_between_two_pixels(f, col - w / 2, 0., col - w / 2, h / 2 - bottom_row)
    delta = angle_between_two_pixels(f, col - w / 2, h / 2 - bottom_row, col - w / 2, h / 2 - top_row)
    Z = rangeZ(h0, gamma, alpha, delta)
    return Z


def YOLOV2(loaded_model, PATH, CameraLogImage, IMAGE_EXTENSION, ROLL, names, colors, sz, TextColorOnOutputImages,
           strokeTextWidth, LineWidth, heights, Focal_lengths, pitch):
    img = cv2.imread(PATH + str(CameraLogImage) + IMAGE_EXTENSION)
    imsz = img.shape[0:2]
    min_open_cv_image_size = min(imsz)
    img = img[0:min_open_cv_image_size, 0:min_open_cv_image_size]
    # get image height, width
    (h, w) = img.shape[:2]
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), (ROLL * 180) / pi, 1.0), (w, h))
    min_open_cv_image_size = min(img.shape[0:2])
    open_cv_image = img[0:min_open_cv_image_size, 0:min_open_cv_image_size]
    open_cv_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
    pil_Image = open_cv_image.resize((sz, sz))
    pred = loaded_model.predict(data={'image': pil_Image})

    textOnScreenX, textOnScreenY = 160, 60
    dist_to_sign = []
    class_numbers = []
    for ix in range(len(pred.get('coordinates'))):
        # create  rectangle image
        xc = pred.get('coordinates')[ix][0] * min_open_cv_image_size  # Center X
        yc = pred.get('coordinates')[ix][1] * min_open_cv_image_size  # Center Y
        w = pred.get('coordinates')[ix][2] * min_open_cv_image_size  # Width
        h = pred.get('coordinates')[ix][3] * min_open_cv_image_size  # Height
        x = xc - (w / 2)  # Top left X
        y = yc - (h / 2)  # Top left Y
        points = ((x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y))
        font = ImageFont.truetype('./FreeMono.ttf', 40)
        draw = ImageDraw.Draw(open_cv_image)
        class_idx = int(np.argmax(pred.get('confidence')[ix]))
        draw.text((xc, yc), str(ix), TextColorOnOutputImages, font=font, stroke_width=strokeTextWidth)
        draw.text((textOnScreenX * 5, textOnScreenY),
                  'Conf: ' + str(pred.get('confidence')[ix][class_idx]),
                  TextColorOnOutputImages, font=font, stroke_width=strokeTextWidth)
        draw.text((textOnScreenX, textOnScreenY), str(ix) + ':  ' + names[class_idx],
                  TextColorOnOutputImages, font=font, stroke_width=strokeTextWidth)
        draw.line(points, fill=colors[class_idx], width=LineWidth)
        textOnScreenY += 60
        # Distance Estimation
        dist_to_sign.append(distance_measuring_new(int(y + h), pitch, heights[int(class_idx)],
                                                   Focal_lengths[0], imsz, y, xc))
        class_numbers.append(class_idx)
    open_cv_image = np.array(open_cv_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image, dist_to_sign, class_numbers


# @jit(nopython=True)
def XYZ_pitch_yaw_roll(imagesName, ARKIT):
    # read the CoreML txt file
    for items in range(0, len(ARKIT)):
        if ARKIT[items].strip() == imagesName.strip():
            [x, y, z] = [float(x) for x in ARKIT[items+4].split(',')]
            [pitch, yaw, roll] = [float(x) for x in ARKIT[items+5].split('(')[1].strip(')').split(',')]
            return x, y, z, pitch, yaw, roll


@jit(nopython=True)
def find_nearest_barrier(binmap, particles_XY, new_particles):
    index_to_remove = []
    h, w = binmap.shape
    for counter in range(0, len(particles_XY)):
        x = particles_XY[counter][Z_]
        y = particles_XY[counter][X_]
        # access image as im[x,y] even though this is not idiomatic!
        # assume that x and y are integers
        c, s = cos(particles_XY[counter][YAW_]), sin(particles_XY[counter][YAW_])
        cnt = 0
        hit = False
        while not hit:
            x2, y2 = round(x + c * cnt), round(y + s * cnt)
            cnt += 1
            if 0 <= x2 < w and 0 <= y2 < h:  # within image borders
                if binmap[y2, x2] > 0:
                    hit = True
            else:
                hit = True  # since we hit the image border
        dist = sqrt((x - x2) ** 2 + (y - y2) ** 2)
        if dist == 0:
            index_to_remove.append(counter)
        else:
            distance = sqrt(((x - new_particles[counter][Z_]) ** 2) + ((y - new_particles[counter][X_]) ** 2))
            if distance > dist:
                index_to_remove.append(counter)
    return index_to_remove


@jit(nopython=True)
def line_draw(startXY, endXY):
    xy = []
    if (endXY[0] - startXY[0]) is not 0:
        slope = (endXY[1] - startXY[1]) / (endXY[0] - startXY[0])
        if slope >= 0:
            # Case in which starting x and y are smaller than end x and y
            if endXY[0] > startXY[0]:
                # In this case we try to estimate to get increment X and find Y or vice versa
                if abs(endXY[0] - startXY[0]) >= abs(endXY[1] - startXY[1]):  # Add X
                    for i in range(startXY[0], endXY[0]):
                        y = startXY[1] + slope * (i - startXY[0])
                        xy.append(tuple((int(round(i)), int(round(y)))))
                        # cv2.circle(canvas, tuple((int(round(i)), int(round(y)))), 1, (128, 255, 255), -1)
                else:
                    for i in range(startXY[1], endXY[1]):
                        x = startXY[0] + (1 / slope) * (i - startXY[1])
                        xy.append(tuple((int(round(x)), int(round(i)))))
                        # cv2.circle(canvas, tuple((int(round(x)), int(round(i)))), 1, (128, 255, 255), -1)
            else:
                if abs(endXY[0] - startXY[0]) >= abs(endXY[1] - startXY[1]):  # Add X
                    for i in range(endXY[0], startXY[0]):
                        y = startXY[1] + slope * (i - startXY[0])
                        xy.append(tuple((int(round(i)), int(round(y)))))
                        # cv2.circle(canvas, tuple((int(round(i)), int(round(y)))), 1, (128, 255, 255), -1)
                else:
                    for i in range(endXY[1], startXY[1]):
                        x = startXY[0] + (1 / slope) * (i - startXY[1])
                        xy.append(tuple((int(round(x)), int(round(i)))))
                        # cv2.circle(canvas, tuple((int(round(x)), int(round(i)))), 1, (128, 255, 255), -1)
        else:
            if endXY[0] < startXY[0]:
                if abs(endXY[0] - startXY[0]) >= abs(endXY[1] - startXY[1]):  # Add X
                    for i in range(endXY[0], startXY[0]):
                        y = startXY[1] + slope * (i - startXY[0])
                        xy.append(tuple((int(round(i)), int(round(y)))))
                        # cv2.circle(canvas, tuple((int(round(i)), int(round(y)))), 1, (128, 255, 255), -1)
                else:
                    for i in range(startXY[1], endXY[1]):
                        x = startXY[0] + (1 / slope) * (i - startXY[1])
                        xy.append(tuple((int(round(x)), int(round(i)))))
                        # cv2.circle(canvas, tuple((int(round(x)), int(round(i)))), 1, (128, 255, 255), -1)
            else:
                if abs(endXY[0] - startXY[0]) >= abs(endXY[1] - startXY[1]):  # Add X
                    for i in range(startXY[0], endXY[0]):
                        y = startXY[1] + slope * (i - startXY[0])
                        xy.append(tuple((int(round(i)), int(round(y)))))
                        # cv2.circle(canvas, tuple((int(round(i)), int(round(y)))), 1, (128, 255, 255), -1)
                else:
                    for i in range(endXY[1], startXY[1]):
                        x = startXY[0] + (1 / slope) * (i - startXY[1])
                        xy.append(tuple((int(round(x)), int(round(i)))))
                        # cv2.circle(canvas, tuple((int(round(x)), int(round(i)))), 1, (128, 255, 255), -1)
    elif endXY[0] - startXY[0] is 0:
        if endXY[1] > startXY[1]:
            for i in range(startXY[1], endXY[1]):
                xy.append(tuple((int(round(startXY[0])), int(round(i)))))
                # cv2.circle(canvas, tuple((int(round(startXY[0])), int(round(i)))), 1, (128, 255, 255), -1)
        else:
            for i in range(endXY[1], startXY[1]):
                xy.append(tuple((int(round(startXY[0])), int(round(i)))))
                # cv2.circle(canvas, tuple((int(round(startXY[0])), int(round(i)))), 1, (128, 255, 255), -1)
    return xy


@jit(nopython=True)
def find_pixels_outside_map(particles, HEIGHT, WIDTH):
    # The following line tries to remove new estimated particles which are out of the size of the image either greater
    # than the height and width or lower than zero
    result = [inx for inx, items in enumerate(particles) if 0 > items[0] or items[0] > WIDTH - 1 or
              0 > items[1] or items[1] > HEIGHT - 1]
    return result


def generate_uniform_particles_data(number_of_particles, image_original, scale, scale_sigma):
    WalkableAreas = np.where(image_original[:, :, 0] == 0)
    WalkableAreas = np.transpose((WalkableAreas[1], WalkableAreas[0])).astype(float)
    return np.hstack((WalkableAreas[np.random.choice(WalkableAreas.shape[0], number_of_particles, replace=True)],
                      np.reshape(np.array([radians(hd) for hd in np.random.uniform(-180, 180, size=number_of_particles)]), (number_of_particles, 1)),
                      np.random.normal(loc=scale, scale=scale_sigma, size=(number_of_particles, 1)),
                      np.ones((number_of_particles, 1), dtype=np.float64)))


def Resampling(particles_data, resampling_threshold, newSize, probability_error, scale, HEIGHT, WIDTH):
    if particles_data.shape[0] <= NumberOfParticles / resampling_threshold:
        index = np.random.choice(particles_data.shape[0], newSize, p=particles_data[:, SCORE_])
        newParticles = particles_data[index]
        newParticles[:, SCORE_] = newParticles[:, SCORE_] * 0 + probability_error
        newParticlesError = np.random.uniform(low=-scale, high=scale, size=(newParticles.shape[0], 2))

        newParticles[:, Z_] = newParticles[:, Z_] + newParticlesError[:, Z_]
        newParticles[:, X_] = newParticles[:, X_] + newParticlesError[:, X_]
        newParticles = newParticles * [1.]
        # The height and width or lower than zero
        FilteredItems = find_pixels_outside_map(newParticles, HEIGHT, WIDTH)
        newParticles = np.array([np.delete(newParticles, FilteredItems[::-1], 0)])[0]
        particles_data = np.concatenate((particles_data, newParticles), axis=0)
        particles_data[:, SCORE_] /= np.sum(particles_data[:, SCORE_])
    return particles_data


def KDE(particles_data, KDE_model, xGrid, yGrid):
    KDEParticles = np.vstack((particles_data[:, Z_], particles_data[:, X_])).T
    KDE_model.fit(KDEParticles)
    heatmap = np.reshape(np.exp(KDE_model.score_samples(np.vstack([xGrid.ravel(), yGrid.ravel()]).T)), xGrid.shape)
    plt.clf()
    plt.contourf(xGrid, yGrid, heatmap, cmap='hot', alpha=.5)


def GaussianHeatmap(particles_data, H, W):
    heatmap = np.zeros(shape=(H, W))
    for p in particles_data:
        r = floor(p[X_]*p[SCALE_])
        c = floor(p[Z_]*p[SCALE_])
        heatmap[r, c] = heatmap[r, c] + p[SCORE_]
    heatmap = cv2.GaussianBlur(heatmap, ksize=(0, 0), sigmaX=10)
    # plt.clf()
    # plt.contourf(heatmap, levels=7, cmap='gnuplot', alpha=.7)
    # Normalize the heatmap and convert data type from float 64 to uint8 for imshow Concatenation purposes
    heatmap /= np.max(np.abs(heatmap))
    heatmap = 255 * heatmap
    return heatmap.astype(np.uint8)


@jit(nopython=True)
def DL_Scoring(scale, Exit_X_Y, particles, E_Map, r):
    xy = []
    for exits_xy in Exit_X_Y:
        mydist = np.sqrt(np.power(particles[:, 0:1] - exits_xy[0], 2) + np.power(particles[:, 1:2] - exits_xy[1], 2))
        inx = 0
        for items in mydist:
            if r - scale < int(round(items[0])) < r + scale:
                if E_Map[int(round(particles[inx, 1:2][0])), int(round(particles[inx, 0:1][0]))] == 255:
                    xy.append(inx)
            inx = inx + 1
    return xy


@jit(nopython=True)
def ShowParticles(particles_data, im_particles):
    num = 0
    for particle in particles_data:
        im_particles[int(round(particle[1])), int(round(particles_data[num, 0]))] = (0, 0, 255)
        num = num + 1
    return im_particles


@jit(nopython=True)
def FilteringParticles(particles_data, imBinary):
    xy = []
    inx = 0
    for particle in particles_data:
        if imBinary[int(round(particle[1])), int(round(particle[0]))] == 255:
            xy.append(inx)
        inx = inx + 1

    return xy


def main(image_original, imgP, number_of_particles, list_of_cameraLog_images, scale, offset_u, offset_v, HEIGHT, WIDTH,
         resampling_threshold, newSize, probability_error, KDE_model, xGrid, yGrid, loaded_model, names, colors, sz,
         TextColorOnOutputImages, strokeTextWidth, LineWidth, sign_heights, focalLength, Exit_X_Y, imBinary, Exits_Map):
    # Create particles located on empty spaces
    particles_data = generate_uniform_particles_data(number_of_particles, image_original, scale, scale*.02)
    previousYAW = 0
    previous_X = 0
    previous_Z = 0
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # leaves no white space around the axes
    # plt.gca().set_axis_off()
    # Start to read each image and its ARKIT Logged data
    for CameraLogImage in list_of_cameraLog_images:
        image_particles = image_original.copy()
        old_particles_data = particles_data.copy()
        ARKIT_DATA = XYZ_pitch_yaw_roll(str(CameraLogImage), ARKIT_LOGGED)
        X_ARKIT = ARKIT_DATA[0] + offset_u / scale
        Z_ARKIT = ARKIT_DATA[2] + offset_v / scale
        Z_ARKIT = -Z_ARKIT
        PITCH = ARKIT_DATA[3]
        YAW = ARKIT_DATA[4]
        ROLL = ARKIT_DATA[5]
        Rot2D_theta = particles_data[:, YAW_] - np.pi / 2 - previousYAW
        DeltaX = X_ARKIT - previous_X
        DeltaZ = Z_ARKIT - previous_Z
        U = DeltaX * np.cos(Rot2D_theta) + DeltaZ * np.sin(Rot2D_theta)
        V = DeltaX * np.sin(Rot2D_theta) - DeltaZ * np.cos(Rot2D_theta)
        particles_data[:, Z_] = particles_data[:, Z_] + U * particles_data[:, SCALE_]
        particles_data[:, X_] = particles_data[:, X_] + V * particles_data[:, SCALE_]

        # This step tries to remove particles that go outside of the indoor space
        FilteredItems = find_pixels_outside_map(particles_data, HEIGHT, WIDTH)
        particles_data = np.array([np.delete(particles_data, FilteredItems[::-1], 0)])[0]
        old_particles_data = np.array([np.delete(old_particles_data, FilteredItems[::-1], 0)])[0]
        # Next step is to remove items that go through walls in indoor space
        FilteredItems = find_nearest_barrier(imgP, old_particles_data, particles_data)
        particles_data = np.array([np.delete(particles_data, FilteredItems[::-1], 0)])[0]
        particles_data[:, SCORE_] /= np.sum(particles_data[:, SCORE_])
        # Resampling step. This step tries to counter number of particles and replace removed particles with new ones.
        particles_data = Resampling(particles_data, resampling_threshold,
                                    newSize, probability_error, scale, HEIGHT, WIDTH)
        # Remove resampled data on walls
        particles_data = np.delete(particles_data, FilteringParticles(particles_data, imBinary), 0)
        # # Update saved data
        particles_data[:, YAW_] = particles_data[:, YAW_] + (YAW - previousYAW)
        # Object detection using YOLO
        object_detection, dist_to_sign, cls_num = YOLOV2(loaded_model, PATH, CameraLogImage, IMAGE_EXTENSION, ROLL,
                                                         names, colors, sz, TextColorOnOutputImages, strokeTextWidth,
                                                         LineWidth, sign_heights, focalLength, PITCH)

        image_particles = ShowParticles(particles_data, image_particles.copy())
        xyz = []
        if 1 in cls_num:
            radius_to_sign = int(round(scale * dist_to_sign[cls_num.index(1)]))
            xyz = DL_Scoring(scale, Exit_X_Y, particles_data, Exits_Map[:, :, 0].copy(), radius_to_sign)

        Signs_LOS_Image = Exits_Map.copy()
        for it in xyz:
            particles_data[it, SCORE_] += 0.1
            Signs_LOS_Image[int(round(particles_data[it, X_])), int(round(particles_data[it, Z_]))] = (0, 0, 255)
        particles_data[:, SCORE_] /= np.sum(particles_data[:, SCORE_])
        HEATMAP = GaussianHeatmap(particles_data, HEIGHT, WIDTH)
        # KDE(particles_data, KDE_model, xGrid, yGrid)
        # -----------------------Visualization Segment------------------------

        # concatanate image Horizontally
        Left_Col = np.concatenate((np.concatenate((Signs_LOS_Image, image_particles), axis=0),
                                  cv2.merge((HEATMAP, HEATMAP, HEATMAP))), axis=0)
        dims = (max(Left_Col.shape), max(Left_Col.shape))
        Right_Col = cv2.resize(object_detection, dims)
        output = np.hstack((Left_Col, Right_Col))
        cv2.imshow("PARTICLES", output)

        # -----------------------Updating Parameters--------------------------
        previousYAW = YAW
        previous_X = X_ARKIT
        previous_Z = Z_ARKIT
        # -----------------------Waiting a little bit to show all windows-----
        if cv2.waitKey(1) & 0xFF == 27:
            break
        plt.pause(0.000001)
        # input("Press Enter to continue...")


if __name__ == "__main__":
    PATH = "../LoggedData/a2/"
    IMAGE_EXTENSION = '.jpg'
    path_to_map = '../maps/walls_4.bmp'
    model_name = '8NMMY2NC15k.mlmodel'
    class_names = ['Safety', 'Exit', 'FaceMask', 'James', 'Caution', 'RedFire', 'Restroom', 'SixFt']
    class_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (192, 192, 192), (192, 0, 0), (0, 192, 0), (0, 0, 192)]
    # FireAlarm: 4.5, FireHose: 38, Sanitizer: 9, A4_Mask: 11, Restroom: 11.75
    Heights = [0.1143, 0.20, 0.28, .20, .15, .18, .13, .06]  # Heights of signs
    EXITS_X_Y = np.array([(94, 47), (68, 178), (278, 178), (292, 157), (392, 157), (392, 85)])
    EXITS_MAP = cv2.imread('./exits_map.bmp')
    FocalLengths = [1602]
    model_input_size = 416
    text_color_output = (0, 0, 0)
    stroke_text_width = 3
    Line_width = 9
    # Number of particles
    NumberOfParticles = 100000
    # This threshold define when to start resampling to speed up the process
    ResamplingThreshold = 5
    NewSizeOfNumberOfParticles = int(NumberOfParticles / 2)
    Scale = 11.7
    Offset_U = 5.7699
    Offset_V = 18.8120
    probabilityError = 1.e-40

    loadedModel = coremltools.models.MLModel('../DeepLearning/TrainedModels/' + model_name)
    # Read Image
    imgOriginal = cv2.imread(path_to_map, 1)
    image_binary = imgOriginal.copy()[:, :, 0]
    image_binary[image_binary > 64] = 255
    image_binary[image_binary <= 64] = 0
    _, imgOriginal = cv2.threshold(imgOriginal, 64, 255, cv2.THRESH_BINARY)
    Im_HEIGHT, Im_WIDTH, dim = imgOriginal.shape
    # make imgOriginal a single-channel grayscale image if it's originally an RGB
    imageP = imgOriginal.copy()[:, :, 0] if len(imgOriginal.shape) == 3 else imgOriginal.copy()
    x_grid, y_grid = np.mgrid[0:Im_WIDTH:25j, 0:Im_HEIGHT:25j]
    # Read the TXT ARKIT data
    ARKIT_LOGGED = [x.strip() for x in open(glob.glob(PATH + "*.txt")[0], "r")]
    # Read Images and sort them
    ListOfCameraLogImages = [int(file.split('/')[len(file.split('/')) - 1].replace(IMAGE_EXTENSION, ''))
                             for file in glob.glob(PATH + "*" + IMAGE_EXTENSION)]
    ListOfCameraLogImages.sort()
    model = KernelDensity(kernel='gaussian', bandwidth=20)

    main(imgOriginal, imageP, NumberOfParticles, ListOfCameraLogImages, Scale, Offset_U, Offset_V, Im_HEIGHT, Im_WIDTH,
         ResamplingThreshold, NewSizeOfNumberOfParticles, probabilityError, model, x_grid, y_grid, loadedModel,
         class_names, class_colors, model_input_size, text_color_output, stroke_text_width, Line_width, Heights,
         FocalLengths, EXITS_X_Y, image_binary, EXITS_MAP)
