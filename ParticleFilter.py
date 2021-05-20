import numpy as np
import cv2
import glob
from numba import jit
from math import sin, cos, sqrt, pi, radians, acos, tan, floor
import matplotlib.pyplot as plt
import coremltools
from PIL import Image, ImageDraw, ImageFont

np.set_printoptions(precision=40)
np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)


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
    min_open_cv_image_size = min(img.shape[0:2])
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
            xyz = ARKIT[items + 4].split(',')
            sp = ARKIT[items + 5].split('<Float>')
            sp1 = sp[1].split(',')
            sp2 = sp1[0].split('(')
            sp3 = sp1[2].split(')')
            pitch = float(sp2[1])
            yaw = float(sp1[1].strip())
            roll = float(sp3[0].strip())
            return float(xyz[0].strip()), float(xyz[1].strip()), float(xyz[2].strip()), pitch, yaw, roll


@jit(nopython=True)
def find_nearest_barrier(binmap, particles_XY, new_particles):
    index_to_remove = []
    h, w = binmap.shape
    for counter in range(0, len(particles_XY)):
        x = particles_XY[counter][0]
        y = particles_XY[counter][1]
        # access image as im[x,y] even though this is not idiomatic!
        # assume that x and y are integers
        c, s = cos(particles_XY[counter][2]), sin(particles_XY[counter][2])
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
            distance = sqrt(((x - new_particles[counter][0]) ** 2) + ((y - new_particles[counter][1]) ** 2))
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


def generate_uniform_particles_data(number_of_particles, image_original, starting_point, R, flag):
    if not flag:
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

    probability_distribution = np.ones((number_of_particles, 1), dtype=np.float64)
    probability_distribution /= np.sum(probability_distribution)
    output_particles = np.hstack((rnd_particles, rnd_yaw, probability_distribution))
    return output_particles


def Resampling(old_particles, newSize):
    index = np.random.choice(old_particles.shape[0], newSize, p=old_particles[:, 3])
    newParticles = old_particles[index] * [1.]
    newParticles[:, 3] = 1
    newParticles[:, 3] /= np.sum(newParticles[:, 3])
    return newParticles


def GaussianHeatmap(particles_data, H, W):
    heatmap = np.zeros(shape=(H, W))
    XY_Columns = np.around(particles_data[:, 0:2]).astype(int)
    heatmap[XY_Columns[:, 1], XY_Columns[:, 0]] += 1
    heatmap = cv2.GaussianBlur(heatmap, ksize=(0, 0), sigmaX=10)
    # plt.clf()
    # plt.contourf(heatmap, levels=7, cmap='gnuplot', alpha=.7)
    # Normalize the heatmap and convert data type from float 64 to uint8 for imshow Concatenation purposes
    heatmap /= np.max(np.abs(heatmap))
    heatmap = 255 * heatmap
    return heatmap.astype(np.uint8)


@jit(nopython=True)
def DL_Scoring(scale, Exit_X_Y, particles, E_Map, r):
    particle_index = []
    for exits_xy in Exit_X_Y:
        mydist = np.sqrt(np.power(particles[:, 0:1] - exits_xy[0], 2) + np.power(particles[:, 1:2] - exits_xy[1], 2))
        inx = 0
        print(r - scale, r + scale)
        for items in mydist:
            if r - scale < int(round(items[0])) < r + scale:
                if E_Map[int(round(particles[inx, 1:2][0])), int(round(particles[inx, 0:1][0]))] == 0:
                    particle_index.append(inx)
            inx = inx + 1
    return particle_index


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


@jit(nopython=True)
def FilteringParticles(particles_data, imBinary):
    xy = []
    inx = 0
    for particle in particles_data:
        if imBinary[int(round(particle[1])), int(round(particle[0]))] == 255:
            xy.append(inx)
        inx = inx + 1

    return xy


def main(image_original, imgP, number_of_particles, list_of_cameraLog_images, scale, HEIGHT, WIDTH,
         loaded_model, names, colors, MLModel_input_size, sign_heights, focalLength, Exit_X_Y, Exits_Map,
         starting_location_flag, user_starting_point, sampling_distance):
    # Create particles located on empty spaces
    particles_data = generate_uniform_particles_data(number_of_particles, image_original, user_starting_point,
                                                     floor(sampling_distance * scale), starting_location_flag)
    imBinary = imgP.copy()
    if starting_location_flag:
        cv2.circle(image_original, center=user_starting_point, color=(0, 255, 0), thickness=-1, radius=3)
    previousYAW = 0
    previous_X = 0
    previous_Z = 0
    h, w, _ = image_original.shape
    # Start to read each image and its ARKIT Logged data
    for CameraLogImage in list_of_cameraLog_images:
        image_particles = image_original.copy()
        old_particles_data = particles_data.copy()
        # Pixels to Meters
        particles_data[:, 0] = particles_data[:, 0] / Scale
        particles_data[:, 1] = (h - particles_data[:, 1]) / Scale
        ARKIT_DATA = XYZ_pitch_yaw_roll(str(CameraLogImage), ARKIT_LOGGED)
        X_ARKIT = ARKIT_DATA[0]
        # Y_ARKIT = ARKIT_DATA[1]
        Z_ARKIT = ARKIT_DATA[2]
        PITCH = ARKIT_DATA[3]
        YAW = ARKIT_DATA[4]
        ROLL = ARKIT_DATA[5]

        Rot2D_theta = particles_data[:, 2] - np.pi/2 - previousYAW
        DeltaX = X_ARKIT - previous_X
        DeltaZ = Z_ARKIT - previous_Z
        U = DeltaX * np.cos(Rot2D_theta) + DeltaZ * np.sin(Rot2D_theta)
        V = DeltaX * np.sin(Rot2D_theta) - DeltaZ * np.cos(Rot2D_theta)

        # Add -1 to 1 meter error to particles X and Y
        XY_ERROR = np.random.uniform(low=-1/(4 * Scale), high=1/(4 * Scale), size=(particles_data.shape[0], 2))
        # Add -3 to 3 degree error to particles Yaw
        YAW_ERROR = np.radians(np.random.uniform(low=-0.25, high=0.25, size=(particles_data.shape[0], 2)))
        # Updating X and Y for each hypothesis
        particles_data[:, 0] = particles_data[:, 0] + U + XY_ERROR[:, 0]
        particles_data[:, 1] = particles_data[:, 1] + V + XY_ERROR[:, 1]
        # Updating Yaw for each hypothesis
        particles_data[:, 2] = particles_data[:, 2] + YAW_ERROR[:, 0]
        # Meters to Pixels
        particles_data[:, 0] = particles_data[:, 0] * Scale
        particles_data[:, 1] = h - particles_data[:, 1] * Scale

        # This step tries to remove particles that go outside of the indoor space
        FilteredItems = find_pixels_outside_map(particles_data, HEIGHT, WIDTH)
        particles_data = np.array([np.delete(particles_data, FilteredItems[::-1], 0)])[0]
        old_particles_data = np.array([np.delete(old_particles_data, FilteredItems[::-1], 0)])[0]
        # This step tries to remove particles that go through walls in indoor space
        FilteredItems = find_nearest_barrier(imgP, old_particles_data, particles_data)
        particles_data = np.array([np.delete(particles_data, FilteredItems[::-1], 0)])[0]
        # Remove particles on walls
        particles_data = np.delete(particles_data, FilteringParticles(particles_data, imBinary), 0)
        # Update YAW data
        particles_data[:, 2] = particles_data[:, 2] + (YAW - previousYAW)
        # Normalize particles probability distribution after deleting particles which go through walls
        particles_data[:, 3] /= np.sum(particles_data[:, 3])
        # Object detection using YOLO
        object_detection, dist_to_sign, cls_num = YOLOV2(loaded_model, PATH, CameraLogImage, IMAGE_EXTENSION, ROLL,
                                                         names, colors, MLModel_input_size, (0, 0, 0), 3, 9,
                                                         sign_heights, focalLength, PITCH)
        LOS_Image = Exits_Map.copy()
        # Find particles on field of view of Exit signs and update their score to them
        if 1 in cls_num:
            radius_to_sign = int(round(scale * dist_to_sign[cls_num.index(1)]))
            Index_Particles_LOS = DL_Scoring(scale, Exit_X_Y, particles_data, Exits_Map[:, :, 0].copy(), radius_to_sign)
            if len(Index_Particles_LOS) is not 0:
                LOS_Image, particles_data = updating_detection_scores(Index_Particles_LOS, particles_data, LOS_Image)
                particles_data = Resampling(particles_data, number_of_particles)
        # -----------------------Updating Parameters--------------------------
        previousYAW = YAW
        previous_X = X_ARKIT
        previous_Z = Z_ARKIT
        # -----------------------Visualization Segment------------------------
        # Show particles on the map
        image_particles = ShowParticles(particles_data, image_particles.copy())
        # Show the heatmap
        HEATMAP = GaussianHeatmap(particles_data, HEIGHT, WIDTH)
        # Concatenate maps for visualization
        Left_Col = np.concatenate((np.concatenate((image_particles, image_particles), axis=0),
                                   cv2.merge((HEATMAP, HEATMAP, HEATMAP))), axis=0)
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
    model_name = '8NMMY2NC15k.mlmodel'
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
    EXITS_X_Y = [(335, 57), (415, 51), (415, 178), (362, 175), (314, 164), (100, 47)]  # 2th Flr
    EXITS_MAP = cv2.bitwise_not(cv2.imread('../LOS_Maps/exits_' + str(flr) + '.bmp'))
    FocalLengths = [1602]
    model_input_size = 416
    # Number of particles
    NumberOfParticles = 10000
    # Read Image
    imgOriginal = cv2.imread(path_to_map, 1)
    img = imgOriginal.copy()
    # show image
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', mouse_click)
    while while_flag:
        cv2.waitKey(2)

    cv2.destroyWindow('image')
    # Sample distance define how far from a selected starting point we need to do sampling
    sample_distance = 2
    Scale = 12
    loadedModel = coremltools.models.MLModel('../DeepLearning/TrainedModels/' + model_name)
    _, imgOriginal = cv2.threshold(imgOriginal, 64, 255, cv2.THRESH_BINARY)
    image_binary = imgOriginal.copy()[:, :, 0]
    Im_HEIGHT, Im_WIDTH, dim = imgOriginal.shape
    # make imgOriginal a single-channel grayscale image if it's originally an RGB
    imageP = imgOriginal.copy()[:, :, 0] if len(imgOriginal.shape) == 3 else imgOriginal.copy()
    # Read the TXT ARKIT data
    ARKIT_LOGGED = [x for x in open(glob.glob(PATH + "*.txt")[0], "r")]
    # Read Images and sort them
    ListOfCameraLogImages = [int(file.split('/')[len(file.split('/')) - 1].replace(IMAGE_EXTENSION, ''))
                             for file in glob.glob(PATH + "*" + IMAGE_EXTENSION)]
    ListOfCameraLogImages.sort()

    main(imgOriginal, imageP, NumberOfParticles, ListOfCameraLogImages, Scale, Im_HEIGHT, Im_WIDTH,
         loadedModel, class_names, class_colors, model_input_size, Heights, FocalLengths, EXITS_X_Y, EXITS_MAP,
         initiated_flag, user_initial_location, sample_distance)
