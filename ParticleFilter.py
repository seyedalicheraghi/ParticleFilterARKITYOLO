import numpy as np
import cv2
import glob
from time import time
from numba import jit
from scipy import ndimage
from math import sin, cos, sqrt, pi, radians
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import coremltools
from PIL import Image, ImageDraw, ImageFont

np.set_printoptions(precision=40)
np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)


def YOLOV2(loaded_model, PATH, CameraLogImage, IMAGE_EXTENSION, ROLL, names,
           colors, sz, TextColorOnOutputImages, strokeTextWidth, LineWidth):
    open_cv_image = ndimage.rotate(cv2.imread(PATH + str(CameraLogImage) + IMAGE_EXTENSION), (ROLL * 180) / pi)
    min_open_cv_image_size = min(open_cv_image.shape[0:2])
    open_cv_image = open_cv_image[0:min_open_cv_image_size, 0:min_open_cv_image_size]
    open_cv_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
    pil_Image = open_cv_image.resize((sz, sz))
    pred = loaded_model.predict(data={'image': pil_Image})

    textOnScreenX, textOnScreenY = 160, 60
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
        draw.text((textOnScreenX * 5, textOnScreenY), 'Conf: ' +
                  str(pred.get('confidence')[ix][class_idx]),
                  TextColorOnOutputImages, font=font, stroke_width=strokeTextWidth)
        draw.text((textOnScreenX, textOnScreenY), str(ix) + ':  ' + names[class_idx],
                  TextColorOnOutputImages, font=font, stroke_width=strokeTextWidth)
        draw.line(points, fill=colors[class_idx], width=LineWidth)
        textOnScreenY += 60
    open_cv_image = np.array(open_cv_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


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
def find_pixels_outside_map(particles, HEIGHT, WIDTH):
    # The following line tries to remove new estimated particles which are out of the size of the image either greater
    # than the height and width or lower than zero
    return [inx for inx, items in enumerate(particles) if 0 > items[0] or items[0] > WIDTH - 1 or
            0 > items[1] or items[1] > HEIGHT - 1]


def generate_uniform_particles_data(number_of_particles, image_original):
    WalkableAreas = np.where(image_original[:, :, 0] == 0)
    WalkableAreas = np.transpose((WalkableAreas[1], WalkableAreas[0])).astype(float)
    return np.hstack((WalkableAreas[np.random.choice(WalkableAreas.shape[0], number_of_particles, replace=True)],
                      np.reshape(
                          np.array([radians(hd) for hd in np.random.uniform(-180, 180, size=number_of_particles)])
                          , (number_of_particles, 1)), np.ones((number_of_particles, 1), dtype=np.float64)))


def Resampling(particles_data, resampling_threshold, newSize, probability_error, scale, HEIGHT, WIDTH):
    if particles_data.shape[0] <= NumberOfParticles / resampling_threshold:
        index = np.random.choice(particles_data.shape[0], newSize, p=particles_data[:, 3])
        newParticles = particles_data[index]
        newParticles[:, 3] = newParticles[:, 3] * 0 + probability_error
        newParticlesError = np.random.uniform(low=-scale, high=scale, size=(newParticles.shape[0], 2))

        newParticles[:, 0] = newParticles[:, 0] + newParticlesError[:, 0]
        newParticles[:, 1] = newParticles[:, 1] + newParticlesError[:, 1]
        newParticles = newParticles * [1.]
        # The height and width or lower than zero
        FilteredItems = find_pixels_outside_map(newParticles, HEIGHT, WIDTH)
        newParticles = np.array([np.delete(newParticles, FilteredItems[::-1], 0)])[0]
        particles_data = np.concatenate((particles_data, newParticles), axis=0)
        particles_data[:, 3] /= np.sum(particles_data[:, 3])
    return particles_data


def KDE(particles_data, KDE_model, xGrid, yGrid):
    KDEParticles = np.vstack((particles_data[:, 0], particles_data[:, 1])).T
    KDE_model.fit(KDEParticles)
    heatmap = np.reshape(np.exp(KDE_model.score_samples(np.vstack([xGrid.ravel(), yGrid.ravel()]).T)), xGrid.shape)
    plt.clf()
    plt.contourf(xGrid, yGrid, heatmap, cmap='hot', alpha=.5)


def main(image_original, imgP, number_of_particles, list_of_cameraLog_images, scale, offset_u, offset_v, HEIGHT, WIDTH,
         resampling_threshold, newSize, probability_error, KDE_model, xGrid, yGrid, loaded_model,
         names, colors, sz, TextColorOnOutputImages, strokeTextWidth, LineWidth):
    # Create particles located on empty spaces
    particles_data = generate_uniform_particles_data(number_of_particles, image_original)
    previousYAW = 0
    previous_X = 0
    previous_Z = 0
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # leaves no white space around the axes
    plt.gca().set_axis_off()
    # Start to read each image and its ARKIT Logged data
    for CameraLogImage in list_of_cameraLog_images:
        image_particles = image_original.copy()
        old_particles_data = particles_data.copy()
        ARKIT_DATA = XYZ_pitch_yaw_roll(str(CameraLogImage), ARKIT_LOGGED)
        X_ARKIT = ARKIT_DATA[0] * scale + offset_u
        Z_ARKIT = ARKIT_DATA[2] * scale + offset_v
        Z_ARKIT = -Z_ARKIT
        YAW = ARKIT_DATA[4]
        ROLL = ARKIT_DATA[5]
        Rot2D_theta = particles_data[:, 2] - np.pi / 2 - previousYAW
        DeltaX = X_ARKIT - previous_X
        DeltaZ = Z_ARKIT - previous_Z
        U = DeltaX * np.cos(Rot2D_theta) + DeltaZ * np.sin(Rot2D_theta)
        V = DeltaX * np.sin(Rot2D_theta) - DeltaZ * np.cos(Rot2D_theta)
        tic = time()
        KDE(particles_data, KDE_model, xGrid, yGrid)
        print('Time per KDE iteration:', (time() - tic))
        plt.imshow(imgOriginal, alpha=.7)
        cv2.imshow("CameraLog", ndimage.rotate(cv2.imread(PATH + str(CameraLogImage) + IMAGE_EXTENSION)[::4, ::4, ],
                                               (ROLL * 180) / pi))
        for num, particle in enumerate(particles_data):
            cv2.circle(image_particles, tuple((particle[0].astype(int), particle[1].astype(int))), 1, (0, 0, 255), -1)
        cv2.imshow("WINDOW_NAME_PARTICLES", image_particles)
        tic = time()
        object_detection = YOLOV2(loaded_model, PATH, CameraLogImage, IMAGE_EXTENSION, ROLL, names,
                                  colors, sz, TextColorOnOutputImages, strokeTextWidth, LineWidth)
        print('Time per YOLOV2 iteration:', (time() - tic))
        cv2.imshow("CameraLog", object_detection)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        plt.pause(0.000001)
        particles_data[:, 0] = particles_data[:, 0] + U
        particles_data[:, 1] = particles_data[:, 1] + V

        tic = time()
        # Second step is to remove particles that go outside of the indoor space
        FilteredItems = find_pixels_outside_map(particles_data, HEIGHT, WIDTH)
        particles_data = np.array([np.delete(particles_data, FilteredItems[::-1], 0)])[0]
        old_particles_data = np.array([np.delete(old_particles_data, FilteredItems[::-1], 0)])[0]
        # Next step is to remove items that go through walls in indoor space
        FilteredItems = find_nearest_barrier(imgP, old_particles_data, particles_data)
        particles_data = np.array([np.delete(particles_data, FilteredItems[::-1], 0)])[0]
        particles_data[:, 3] /= np.sum(particles_data[:, 3])
        # Resampling step. This step tries to counter number of particles and replace removed particles with new ones.
        particles_data = Resampling(particles_data, resampling_threshold,
                                    newSize, probability_error, scale, HEIGHT, WIDTH)
        # # Update saved data
        particles_data[:, 2] = particles_data[:, 2] + (YAW - previousYAW)
        previousYAW = YAW
        previous_X = X_ARKIT
        previous_Z = Z_ARKIT
        print('Time per Particle Filter iteration:', (time() - tic))
        print("-----------------------", particles_data.shape[0], "---------------------------")


if __name__ == "__main__":
    PATH = "./LoggedData/a2/"
    IMAGE_EXTENSION = '.jpg'
    path_to_map = './maps/walls_4.bmp'
    model_name = '8NMMY2NC15t.mlmodel'
    class_names = ['Safety', 'Exit', 'FaceMask', 'James', 'Caution', 'RedFire', 'Restroom', 'SixFt']
    class_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (192, 192, 192), (192, 0, 0), (0, 192, 0), (0, 0, 192)]
    model_input_size = 416
    text_color_output = (0, 0, 0)
    stroke_text_width = 3
    Line_width = 9
    # Number of particles
    NumberOfParticles = 100000
    # This threshold define when to start resampling to speed up the process
    ResamplingThreshold = 5
    NewSizeOfNumberOfParticles = int(NumberOfParticles / 2)
    Scale = 11.23
    Offset_U = 5.7699
    Offset_V = 18.8120
    probabilityError = 1.e-40
    loadedModel = coremltools.models.MLModel('../DeepLearning/TrainedModels/' + model_name)
    # Read Image
    imgOriginal = cv2.imread(path_to_map, 1)
    _, imgOriginal = cv2.threshold(imgOriginal, 64, 255, cv2.THRESH_BINARY)
    Im_HEIGHT, Im_WIDTH, dim = imgOriginal.shape
    # make imgOriginal a single-channel grayscale image if it's originally an RGB
    imageP = imgOriginal.copy()[:, :, 0] if len(imgOriginal.shape) == 3 else imgOriginal.copy()
    x_grid, y_grid = np.mgrid[0:Im_WIDTH:25j, 0:Im_HEIGHT:25j]
    # Read the TXT ARKIT data
    ARKIT_LOGGED = [x for x in open(glob.glob(PATH + "*.txt")[0], "r")]
    # Read Images and sort them
    ListOfCameraLogImages = [int(file.split('/')[len(file.split('/')) - 1].replace(IMAGE_EXTENSION, ''))
                             for file in glob.glob(PATH + "*" + IMAGE_EXTENSION)]
    ListOfCameraLogImages.sort()
    model = KernelDensity(kernel='gaussian', bandwidth=20)

    main(imgOriginal, imageP, NumberOfParticles, ListOfCameraLogImages, Scale, Offset_U, Offset_V, Im_HEIGHT, Im_WIDTH,
         ResamplingThreshold, NewSizeOfNumberOfParticles, probabilityError, model, x_grid, y_grid, loadedModel,
         class_names, class_colors, model_input_size, text_color_output, stroke_text_width, Line_width)
