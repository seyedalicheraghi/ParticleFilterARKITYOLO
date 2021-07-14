import numpy as np
import cv2
import glob
from numba import jit
from math import sin, cos, sqrt, pi, radians, acos, tan, floor
import matplotlib.pyplot as plt
import coremltools
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk

np.set_printoptions(precision=40)
np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)

# Constant variables to describe particle data array index
Z_ = 0
X_ = 1
YAW_ = 2
SCALE_ = 4
SCORE_ = 3


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
        if class_idx is 1 and pred.get('confidence')[ix][class_idx] > 0.6:
            draw.text((xc, yc), str(ix), TextColorOnOutputImages, font=font, stroke_width=strokeTextWidth)
            # draw.text((textOnScreenX * 5, textOnScreenY),
            #           'Conf: ' + str(pred.get('confidence')[ix][class_idx]),
            #           TextColorOnOutputImages, font=font, stroke_width=strokeTextWidth)

            draw.text((textOnScreenX, textOnScreenY), str(ix) + ':  ' + names[class_idx],
                  TextColorOnOutputImages, font=font, stroke_width=strokeTextWidth)
            draw.line(points, fill=colors[class_idx], width=LineWidth)
            # textOnScreenY += 60
            # Distance Estimation
            dis = distance_measuring_new(int(y + h), pitch, heights[int(class_idx)], Focal_lengths[0], imsz, y, xc)
            dist_to_sign.append(dis)
            draw.text((textOnScreenX * 5, textOnScreenY), 'Dist: ' + str(dis),
                      TextColorOnOutputImages, font=font, stroke_width=strokeTextWidth)
            class_numbers.append(class_idx)
            # print(dis, pred.get('confidence')[ix][class_idx])
    open_cv_image = np.array(open_cv_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image, dist_to_sign, class_numbers


# @jit(nopython=True)
def XYZ_pitch_yaw_roll(imagesName, ARKIT):
    # read the CoreML txt file
    for items in range(0, len(ARKIT)):
        if ARKIT[items].strip() == imagesName.strip():
            [x, y, z] = [float(x) for x in ARKIT[items + 4].split(',')]
            [pitch, yaw, roll] = [float(x) for x in ARKIT[items + 5].split('(')[1].strip(')').split(',')]
            return x, y, z, pitch, yaw, roll


@jit(nopython=True)
def findInvalidSteps(wallmap, oldParticles, newParticles):
    HEIGHT, WIDTH = wallmap.shape
    index_to_remove = []
    # Iterate through each particle to see if it is outside boundaries or crosses a wall
    for idx in range(len(newParticles)):
        oldRow = floor(oldParticles[idx, X_])
        newRow = floor(newParticles[idx, X_])
        oldCol = floor(oldParticles[idx, Z_])
        newCol = floor(newParticles[idx, Z_])
        if 0 > newCol or newCol > WIDTH - 1 or 0 > newRow or newRow > HEIGHT - 1:
            index_to_remove.append(idx)
            continue
        pixsteps = round(max(abs(newRow - oldRow), abs(newCol - oldCol)))
        safe_denom = max(pixsteps, 1)
        for step in range(pixsteps + 1):
            Row = int(oldRow + ((newRow - oldRow) / safe_denom) * step)
            Col = int(oldCol + ((newCol - oldCol) / safe_denom) * step)
            if wallmap[Row, Col] > 0:
                index_to_remove.append(idx)
                break
    # #DISABLED DUE TO NUMBA COMPATIBILITY
    #        Xs = np.linspace(oldPos[X_], newPos[X_], pixsteps, dtype=int)
    #        Zs = np.linspace(oldPos[Z_], newPos[Z_], pixsteps, dtype=int)
    #        if sum(wallmap[Xs, Zs]) > 0:
    #            index_to_remove.append(idx)
    return index_to_remove


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
            xnew = floor(new_particles[counter][Z_])
            ynew = floor(new_particles[counter][X_])
            distance = sqrt(((x - xnew) ** 2) + ((y - ynew) ** 2))
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
    result = [inx for inx, items in enumerate(particles) if 0 > items[Z_] or items[Z_] > WIDTH - 1 or
              0 > items[X_] or items[X_] > HEIGHT - 1]
    return result


def generate_uniform_particles_data(number_of_particles, image_original, starting_point, R, start_flag, yawflag):
    if not start_flag:
        WalkableAreas = np.where(image_original[:, :, 0] == 0)
        WalkableAreas = np.transpose((WalkableAreas[1], WalkableAreas[0])).astype(float)
        rnd_indexes = np.random.choice(WalkableAreas.shape[0], number_of_particles, replace=True)
        rnd_particles = WalkableAreas[rnd_indexes]
    else:
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
                         (number_of_particles, 1))
    if yawflag >= 0:
        rnd_yaw = rnd_yaw * 0 + yawflag
        print(f"Setting initial yaw to {yawflag}.\n")
    probability_distribution = np.ones((number_of_particles, 1), dtype=np.float64)
    probability_distribution /= np.sum(probability_distribution)
    output_particles = np.hstack((rnd_particles, rnd_yaw, probability_distribution))
    return output_particles


def Resampling(old_particles, newSize):
    index = np.random.choice(old_particles.shape[0], newSize, p=old_particles[:, SCORE_])
    newParticles = old_particles[index] * [1.]
    newParticles[:, SCORE_] = float(newSize) ** -1
    return newParticles


def GaussianHeatmap(particles_data, H, W):
    heatmap = np.zeros(shape=(H, W))
    for p in particles_data:
        r = floor(p[X_])
        c = floor(p[Z_])
        heatmap[r, c] = heatmap[r, c] + p[SCORE_]
    heatmap = cv2.GaussianBlur(heatmap, ksize=(0, 0), sigmaX=10)
    # plt.clf()
    # plt.contourf(heatmap, levels=7, cmap='gnuplot', alpha=.7)
    # Normalize the heatmap and convert data type from float 64 to uint8 for imshow Concatenation purposes
    heatmap /= np.max(np.abs(heatmap))
    heatmap = 255 * heatmap
    return heatmap.astype(np.uint8)


@jit(nopython=True)
def PeakCandidates(pm, H, W):
    peakCandidates = []
    entropy = 0.0
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if ((pm[i][j] >= pm[i - 1][j - 1]) and (pm[i][j] >= pm[i - 1][j]) and (pm[i][j] >= pm[i - 1][j + 1]) and
                    (pm[i][j] >= pm[i + 0][j - 1]) and (pm[i][j] >= 0) and (pm[i][j] >= pm[i + 0][j + 1]) and
                    (pm[i][j] >= pm[i + 1][j - 1]) and (pm[i][j] >= pm[i + 1][j]) and (
                            pm[i][j] >= pm[i + 1][j + 1]) and (
                            (pm[i][j] > pm[i - 1][j - 1]) or (pm[i][j] > pm[i - 1][j]) or (
                            pm[i][j] > pm[i - 1][j + 1]) or
                            (pm[i][j] > pm[i + 0][j - 1]) or (pm[i][j] > pm[i + 0][j + 1]) or
                            (pm[i][j] > pm[i + 1][j - 1]) or (pm[i][j] > pm[i + 1][j]) or (
                                    pm[i][j] > pm[i + 1][j + 1]))):
                peakCandidates.append((pm[i][j], i, j))
        entropy -= pm[i][j] * np.log(pm[i][j])
    return peakCandidates, entropy


def ComputePeaks(particles_data, H, W, distTresh=10, nPeaks=2):
    peakmap = np.zeros(shape=(H, W))
    # Add each particle's score at the corresponding map pixel location
    for p in particles_data:
        x = floor(p[X_])
        y = floor(p[Z_])
        peakmap[x, y] = peakmap[x, y] + p[SCORE_]
    # Blur the image to merge nearby hypotheses
    peakmap = cv2.GaussianBlur(peakmap, ksize=(0, 0), sigmaX=10)
    peakCandidates, entropy = PeakCandidates(peakmap, H, W)
    peakCandidates.sort(reverse=True, key=lambda x: x[0])
    peakList = []
    for candidate in peakCandidates:
        peakOK = True
        for peak in peakList:
            dist = sqrt((candidate[1] - peak[1]) ** 2 + (candidate[2] - peak[2]) ** 2)
            if dist < distTresh:
                peakOK = False
                break
        if peakOK: peakList.append(candidate)
        if len(peakList) >= nPeaks: break
    return peakList, entropy


@jit(nopython=True)
def DL_Scoring(Exit_X_Y, particles, EXITS_ORI, EXITS_TH, E_Map, min_r, max_r, T_Facing):
    particle_index = []
    exits_index = []
    cc = 0
    for exits_xy in Exit_X_Y:
        RX = particles[:, 0:1] - exits_xy[0]
        RY = exits_xy[1] - particles[:, 1:2]
        R_Norms = np.sqrt(np.power(RX, 2) + np.power(RY, 2))
        CS = np.cos(particles[:, 2])
        SN = np.sin(particles[:, 2])
        inx = 0
        for R_Norm, Rx, Ry, cs, sn in zip(R_Norms, RX, RY, CS, SN):
            if min_r < int(round(R_Norm[0])) < max_r:
                Camera_Facing = (cs * -Rx[0] + sn * -Ry[0]) / R_Norm
                # Camera_Facing = np.dot([cs, sn], [EXITS_ORI[cc][0], EXITS_ORI[cc][1]])
                # Front_Sign = np.dot([Rx[0], Ry[0]], [EXITS_ORI[cc][0], EXITS_ORI[cc][1]])/R_Norm
                Front_Sign = (Rx[0] * EXITS_ORI[cc][0] + Ry[0] * EXITS_ORI[cc][1]) / R_Norm
                if Front_Sign > EXITS_TH[cc] and Camera_Facing >= T_Facing:
                    # print(Camera_Facing, Front_Sign, [EXITS_ORI[cc][0], EXITS_ORI[cc][1]], cs, sn)
                    # if E_Map[int(round(particles[inx, 1:2][0])), int(round(particles[inx, 0:1][0]))] == 0:
                    particle_index.append(inx)
                    exits_index.append(cc)
            inx = inx + 1
        cc = cc + 1
    particles_res = []
    exits_res = []
    [particles_res.append(x) for x in particle_index if x not in particles_res]
    [exits_res.append(x) for x in exits_index if x not in exits_res]
    return particles_res, exits_res


@jit(nopython=True)
def updating_detection_scores(Particles_in_LOS_Index, p_data, IMAGE_LOS):
    # Find particles predicted to see the sign and give higher probability to them
    for inx in Particles_in_LOS_Index:
        p_data[inx, 3] *= 2
        # For Visualization purposes
        IMAGE_LOS[int(round(p_data[inx, 1])), int(round(p_data[inx, 0]))] = (0, 0, 255)
    # Normalize particles probability distribution
    p_data[:, 3] /= np.sum(p_data[:, 3])
    return IMAGE_LOS, p_data


@jit(nopython=True)
def ShowParticles(particles_data, im_particles):
    for particle in particles_data:
        im_particles[int(round(particle[1])), int(round(particle[0]))] = (0, 0, 255)
    return im_particles


def ShowParticlesYaw(particles_data, im_particles):
    cmap = cv2.applyColorMap(np.array(range(256), np.uint8), cv2.COLORMAP_HSV)
    for particle in particles_data:
        color = cmap[round(particle[YAW_]*128/np.pi) % 256][0]
        im_particles[int(round(particle[1])), int(round(particle[0]))] = color
    return im_particles


def ShowPeaks(peakList, im_peak, nPeaks=1, color=(0, 255, 0)):
    for i in range(min(nPeaks, len(peakList))):
        x, y = floor(peakList[i][1]), floor(peakList[i][2])
        im_peak[x - 1, y] = color
        im_peak[x, y] = color
        im_peak[x + 1, y] = color
        im_peak[x, y - 1] = color
        im_peak[x, y + 1] = color

    return im_peak


def DrawPeaks(peakList, im_peak, nPeaks=1):
    imgPeak = Image.fromarray(im_peak)
    drawpeak = ImageDraw.Draw(imgPeak)
    for i in range(min(nPeaks, len(peakList))):
        x, y = floor(peakList[i][2]), floor(peakList[i][1])
        drawpeak.line((x, y - 1, x, y + 1), fill=(0, 255, 0), width=1)
        drawpeak.line((x - 1, y, x + 1, y), fill=(0, 255, 0), width=1)
    return im_peak


@jit(nopython=True)
def FilteringParticles(particles_data, imBinary):
    xy = []
    inx = 0
    for particle in particles_data:
        if imBinary[int(round(particle[1])), int(round(particle[0]))] == 255:
            xy.append(inx)
        inx = inx + 1

    return xy


def main(image_original, imgP, number_of_particles, list_of_cameraLog_images, scale, HEIGHT, WIDTH, loaded_model, names,
         colors, MLModel_input_size, sign_heights, focalLength, Exit_X_Y, Exits_Map, starting_location_flag,
         user_starting_point, sampling_distance, ADDED_ERROR, EXITS_ORI, EXITS_TH, plot_angles, TH_Facing, yaw_flag=-1,
         radseed=542014):
    # Create particles located on empty spaces
    np.random.seed(radseed)
    particles_data = generate_uniform_particles_data(number_of_particles, image_original, user_starting_point,
                                                     floor(sampling_distance * scale), starting_location_flag, yaw_flag)
    imBinary = imgP.copy()
    if starting_location_flag:
        cv2.circle(image_original, center=user_starting_point, color=(0, 255, 0), thickness=-1, radius=3)
    previousYAW = 0
    previous_X = 0
    previous_Z = 0
    h, w, _ = image_original.shape
    Initializing_Flag = False
    trajectory = []
    # Start to read each image and its ARKIT Logged data
    for CameraLogImage in list_of_cameraLog_images:
        image_particles = image_original.copy()
        radius_image = image_original.copy()
        old_particles_data = particles_data.copy()
        # Pixels to Meters
        particles_data[:, Z_] = particles_data[:, Z_] / Scale
        particles_data[:, X_] = (h - particles_data[:, X_]) / Scale
        ARKIT_DATA = XYZ_pitch_yaw_roll(str(CameraLogImage), ARKIT_LOGGED)
        X_ARKIT = ARKIT_DATA[0]
        # Y_ARKIT = ARKIT_DATA[1]
        Z_ARKIT = ARKIT_DATA[2]
        PITCH = ARKIT_DATA[3]
        YAW = ARKIT_DATA[4]
        ROLL = ARKIT_DATA[5]
        if not Initializing_Flag:
            previous_X = X_ARKIT
            previous_Z = Z_ARKIT
            Initializing_Flag = True
            previousYAW = YAW
        Rot2D_theta = particles_data[:, YAW_] - np.pi / 2 - previousYAW
        DeltaX = X_ARKIT - previous_X
        DeltaZ = Z_ARKIT - previous_Z
        U = DeltaX * np.cos(Rot2D_theta) + DeltaZ * np.sin(Rot2D_theta)
        V = DeltaX * np.sin(Rot2D_theta) - DeltaZ * np.cos(Rot2D_theta)

        # Add -1 to 1 meter error to particles X and Y
        XY_ERROR = np.random.uniform(low=-2 / (4 * Scale), high=2 / (4 * Scale), size=(particles_data.shape[0], 2))
        # Add -3 to 3 degree error to particles Yaw
        YAW_ERROR = np.radians(np.random.uniform(low=-0.25, high=0.25, size=(particles_data.shape[0], 2)))
        if ADDED_ERROR is 0:
            XY_ERROR *= 0
            YAW_ERROR *= 0
        # Updating X and Y for each hypothesis
        particles_data[:, Z_] = particles_data[:, Z_] + U + XY_ERROR[:, Z_]
        particles_data[:, X_] = particles_data[:, X_] + V + XY_ERROR[:, X_]
        # Updating Yaw for each hypothesis
        particles_data[:, YAW_] = particles_data[:, YAW_] + YAW_ERROR[:, Z_]
        # Meters to Pixels
        particles_data[:, Z_] = particles_data[:, Z_] * Scale
        particles_data[:, X_] = h - particles_data[:, X_] * Scale

        # Remove particles landing outside the space or stepping through a wall
        particles_data = np.delete(particles_data, findInvalidSteps(imBinary, old_particles_data, particles_data), 0)
        particles_data = np.delete(particles_data, FilteringParticles(particles_data, imBinary), 0)

        # Update YAW data
        particles_data[:, YAW_] = particles_data[:, YAW_] + (YAW - previousYAW)
        # Normalize particles probability distribution after deleting particles which go through walls
        particles_data[:, SCORE_] /= np.sum(particles_data[:, SCORE_])
        # Object detection using YOLO
        object_detection, dist_to_sign, cls_num = YOLOV2(loaded_model, PATH, CameraLogImage, IMAGE_EXTENSION, ROLL,
                                                         names, colors, MLModel_input_size, (0, 0, 0), 3, 9,
                                                         sign_heights, focalLength, PITCH)

        # Find particles on field of view of Exit signs and update their score to them
        for cc, ex in enumerate(Exit_X_Y):
            cv2.ellipse(radius_image, (ex[0], ex[1]), (30, 30),
                        0, int(round(plot_angles[cc][0])), int(round(plot_angles[cc][1])), (255, 0, 0), 10)
            cv2.circle(radius_image, center=(ex[0], ex[1]), color=(255, 255, 0), thickness=-1, radius=3)
        if 1 in cls_num:
            radius_to_sign = int(round(scale * dist_to_sign[cls_num.index(1)]))
            min_radius = int(round(radius_to_sign - 2 * scale))
            max_radius = int(round(radius_to_sign + 2 * scale))
            Index_Particles_LOS, closest_exits = DL_Scoring(Exit_X_Y, particles_data, EXITS_ORI, EXITS_TH,
                                                            Exits_Map[:, :, 0].copy(), min_radius, max_radius,
                                                            TH_Facing)
            for ex in closest_exits:
                cl = (255, 0, 0)
                if min_radius < 0:
                    min_radius = 0
                cv2.ellipse(image_particles, (Exit_X_Y[ex][0], Exit_X_Y[ex][1]), (radius_to_sign, radius_to_sign),
                            0, int(round(plot_angles[ex][0])), int(round(plot_angles[ex][1])), cl,
                            int(round(max_radius - min_radius)))
            if len(Index_Particles_LOS) is not 0:
                # Find particles predicted to see the sign and give higher probability to them
                particles_data[Index_Particles_LOS, 3] *= 2
                # Normalize particles probability distribution
                particles_data[:, SCORE_] /= np.sum(particles_data[:, SCORE_])
            if len(particles_data) is not 0:
                particles_data = Resampling(particles_data, number_of_particles)
            else:
                particles_data = generate_uniform_particles_data(number_of_particles, image_original,
                                                                 user_starting_point, floor(sampling_distance * scale),
                                                                 starting_location_flag, yaw_flag)
        # -----------------------Updating Parameters--------------------------
        previousYAW = YAW
        previous_X = X_ARKIT
        previous_Z = Z_ARKIT
        # -----------------------Visualization Segment------------------------
        # Show particles on the map
        image_particles = ShowParticlesYaw(particles_data, image_particles.copy())
        for ex in Exit_X_Y:
            cv2.circle(image_particles, center=(ex[0], ex[1]), color=(255, 255, 0), thickness=-1, radius=3)
        # Find peaks
        peakList, entropy = ComputePeaks(particles_data, HEIGHT, WIDTH, distTresh=12, nPeaks=30)
        if len(peakList) < 1:
            trajectory.append([0, 0])
        else:
            trajectory.append(peakList[0])

        # Generate the heat map for display
        HEATMAP = GaussianHeatmap(particles_data, HEIGHT, WIDTH)
        image_heatmap = cv2.merge((HEATMAP, HEATMAP, HEATMAP))

        image_particles = ShowPeaks(peakList, image_particles, 1)
        image_heatmap = ShowPeaks(peakList, image_heatmap, 100)
        image_heatmap = ShowPeaks(peakList, image_heatmap, 1, (0, 0, 255))

        # Concatenate maps for visualization
        Left_Col = np.concatenate((np.concatenate((image_particles, radius_image), axis=0), image_heatmap), axis=0)
        dims = (max(Left_Col.shape), max(Left_Col.shape))
        Right_Col = cv2.resize(object_detection, dims)
        output = np.hstack((Left_Col, Right_Col))
        cv2.imshow("PARTICLES", output)
        cv2.setMouseCallback('PARTICLES', mouse_click)
        while while_flag:
            cv2.waitKey(1)
        # -----------------------Waiting a little bit to show all windows-----
        if cv2.waitKey(1) & 0xFF == 27:
            break
        plt.pause(0.000001)
    return trajectory


img = []
while_flag = True
# Change the flag to change modes between known or unknown starting location
initiated_flag = True
# This flag is responsible to pause and continue the process
control_flag = False
# This section tries to assign starting location of the user
user_initial_location = (94, 47)
fields = ['Floor Number', 'Number of particles', 'Frame Number', 'Initial YAW as S, N, ...', 'Adding Error (0 or 1)',
          "Scale", "Sampling Distance ", "Exit Front Th", "Exit Facing Th", "Number of Trials", "Model name"]


def makeform(root, fields):
    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=15, text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries


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
            print(f"Initial location at: ({x}, {y}).\n")
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
    root = tk.Tk()
    entries = makeform(root, fields)
    b2 = tk.Button(root, text='Start', command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()
    FocalLengths = [1602]
    model_input_size = 416
    randseed = 542014
    initial_user_data = [4, 50000, 0, -1, 1, 11.23, 2, 0.1, 0.1, 1, '8NMMY2NC15k']
    Directions = {"E": 0, "N": np.pi / 2, "W": np.pi, "S": 3 * (np.pi / 2)}
    for count, entry in enumerate(entries):
        field = entry[0]
        text = entry[1].get()
        print('%s: "%s"' % (field, text))
        if len(text) > 0:
            if count is 3:
                initial_user_data[count] = Directions[text]
            elif count is 10:
                initial_user_data[count] = text
            else:
                initial_user_data[count] = float(text)

    flr = int(round(initial_user_data[0]))
    # Number of particles
    NumberOfParticles = int(round(initial_user_data[1]))
    FRAME_TO_START = int(round(initial_user_data[2]))
    PARTICLES_YAW = initial_user_data[3]
    Additional_Error = int(round(initial_user_data[4]))
    Scale = initial_user_data[5]
    # Sample distance define how far from a selected starting point we need to do sampling
    sample_distance = int(round(initial_user_data[6]))
    TH_Front = initial_user_data[7]
    TH_Facing = initial_user_data[8]
    # test harness loop
    ntrials =  int(initial_user_data[9])
    model_name = initial_user_data[10] + '.mlmodel'
    PATH = "../LoggedData/trial" + str(flr) + "/"
    IMAGE_EXTENSION = '.jpg'
    path_to_map = '../maps/walls_' + str(flr) + '.bmp'
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
    # EXITS_X_Y = [(321, 50), (395, 86), (394, 158), (293, 159), (279, 177), (71, 177), (73, 52), (100, 48)]  # 4th Flr
    # EXITS_X_Y = [(99, 44), (72, 51), (71, 190), (270, 174), (421, 188), (421, 49)]  # 3th Flr

    # 2th Flr
    EXITS_X_Y = [(100, 47), (335, 57), (335, 57), (415, 51), (415, 178), (362, 175), (362, 175), (314, 164), (314, 164)]
    EXITS_DIR = [(1, 0), (1, 0), (-1, 0), (0, -1), (0, 1), (1, 0), (-1, 0), (0, -1), (0, 1)]
    EXITS_TH = [TH_Front] * len(EXITS_X_Y)
    plot_angles = []
    for items_dir, items_th in zip(EXITS_DIR, EXITS_TH):
        angle_to_cover = 180 - 180 * items_th
        if items_dir is (1, 0):
            starting_angle = 270 + 90 * items_th
            plot_angles.append((starting_angle, starting_angle + angle_to_cover))
        elif items_dir is (-1, 0):
            starting_angle = 90 + 90 * items_th
            plot_angles.append((starting_angle, starting_angle + angle_to_cover))
        elif items_dir is (0, -1):
            starting_angle = 0 + 90 * items_th
            plot_angles.append((starting_angle, starting_angle + angle_to_cover))
        elif items_dir is (0, 1):
            starting_angle = 180 + 90 * items_th
            plot_angles.append((starting_angle, starting_angle + angle_to_cover))
    EXITS_MAP = cv2.bitwise_not(cv2.imread('../LOS_Maps/exits_' + str(flr) + '.bmp'))
    # Read Image
    imgOriginal = cv2.imread(path_to_map, 1)
    img = imgOriginal.copy()
    # show image
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', mouse_click)
    while while_flag:
        cv2.waitKey(2)

    cv2.destroyWindow('image')
    loadedModel = coremltools.models.MLModel('../DeepLearning/TrainedModels/' + model_name)
    _, imgOriginal = cv2.threshold(imgOriginal, 64, 255, cv2.THRESH_BINARY)
    image_binary = imgOriginal.copy()[:, :, 0]
    Im_HEIGHT, Im_WIDTH, dim = imgOriginal.shape
    # make imgOriginal a single-channel grayscale image if it's originally an RGB
    imageP = imgOriginal.copy()[:, :, 0] if len(imgOriginal.shape) == 3 else imgOriginal.copy()
    # Read the TXT ARKIT data
    ARKIT_LOGGED = [x.strip() for x in open(glob.glob(PATH + "*.txt")[0], "r")]
    # Read Images and sort them
    ListOfCameraLogImages = [int(file.split('/')[len(file.split('/')) - 1].replace(IMAGE_EXTENSION, ''))
                             for file in glob.glob(PATH + "*" + IMAGE_EXTENSION)]
    ListOfCameraLogImages.sort()
    if FRAME_TO_START > 1:
        ListOfCameraLogImages = ListOfCameraLogImages[FRAME_TO_START - 1:]
    # Build Ground Truth
    gt = main(imgOriginal, imageP, 10000, ListOfCameraLogImages, Scale, Im_HEIGHT, Im_WIDTH, loadedModel, class_names,
              class_colors, model_input_size, Heights, FocalLengths, EXITS_X_Y, EXITS_MAP, initiated_flag,
              user_initial_location, sample_distance, Additional_Error, EXITS_DIR, EXITS_TH, plot_angles, TH_Facing,
              PARTICLES_YAW, randseed)

    trajectoryDistThresh = 1  # meters away from ground truth peaks must be to count correct
    convergetime = []
    np.random.seed(randseed)
    randseeds = (np.random.randint(low=1, high=1000000, size=ntrials))
    for trial in range(ntrials):
        peakpath = main(imgOriginal, imageP, NumberOfParticles, ListOfCameraLogImages, Scale, Im_HEIGHT, Im_WIDTH,
                        loadedModel, class_names, class_colors, model_input_size, Heights, FocalLengths, EXITS_X_Y,
                        EXITS_MAP, False, user_initial_location, sample_distance, Additional_Error, EXITS_DIR, EXITS_TH,
                        plot_angles, TH_Facing, PARTICLES_YAW, randseeds[trial])
        for step in reversed(range(len(peakpath))):
            dist = sqrt((gt[step][0] - peakpath[step][0]) ** 2 + (gt[step][1] - peakpath[step][1]) ** 2)
            if dist > Scale * trajectoryDistThresh:
                convergetime.append(step)
                break
    meanconv = sum(convergetime) / len(convergetime)
    print(f"Over {ntrials} trials, the average time for peak to converge correctly is {meanconv}.\n")
    print(f"It converged {len(convergetime)} times out of {ntrials} trials.\n")
