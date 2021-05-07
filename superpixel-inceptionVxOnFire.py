################################################################################

# Example : perform live fire detection in video using superpixel localization
# and the superpixel trained version of the InceptionV1-OnFire,
# InceptionV3-OnFire and InceptionV4-OnFire CNN models

# Copyright (c) 2017/18 - Andrew Dunnings / Toby Breckon, Durham University, UK
# Copyright (c) 2019/20 - Ganesh Samarth / Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

################################################################################

import cv2
import os
import sys
import math
import numpy as np
import argparse

################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

from inceptionVxOnFire import construct_inceptionv1onfire, construct_inceptionv3onfire, construct_inceptionv4onfire

################################################################################

def extract_bounded_nonzero(input):

    gray = input[:, :, 0];

    rows = np.any(gray, axis=1)
    cols = np.any(gray, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return input[cmin:cmax,rmin:rmax]


def pad_image(image, new_width, new_height, pad_value = 0):

    padded = np.zeros((new_width, new_height, image.shape[2]), dtype=np.uint8)

    pos_x = int(np.round((new_width / 2) - (image.shape[1] / 2)))
    pos_y = int(np.round((new_height / 2) - (image.shape[0] / 2)))

    padded[pos_y:image.shape[0]+pos_y,pos_x:image.shape[1]+pos_x] = image

    return padded

################################################################################

parser = argparse.ArgumentParser(description='Perform superpixel based InceptionV1/V3/V4 fire detection on incoming video')
parser.add_argument("-m", "--model_to_use", type=int, help="specify model to use", default=1, choices={1, 3, 4})
parser.add_argument('video_file', metavar='video_file', type=str, help='specify video file')
args = parser.parse_args()

print("Constructing SP-InceptionV" + str(args.model_to_use) + "-OnFire ...")

if (args.model_to_use == 1):

    # use InceptionV1-OnFire CNN model - [Dunning/Breckon, 2018]
    model = construct_inceptionv1onfire (224, 224, training=False)
    model.load(os.path.join("models/SP-InceptionV1-OnFire", "sp-inceptiononv1onfire"),weights_only=True)

elif (args.model_to_use == 3):

    model = construct_inceptionv3onfire (224, 224, training=False)
    model.load(os.path.join("models/SP-InceptionV3-OnFire", "sp-inceptionv3onfire"),weights_only=False)

elif (args.model_to_use == 4):

    model = construct_inceptionv4onfire (224, 224, training=False)
    model.load(os.path.join("models/SP-InceptionV4-OnFire", "sp-inceptionv4onfire"),weights_only=False)

print("Loaded CNN network weights ...")

################################################################################

rows = 224
cols = 224

windowName = "Live Fire Detection - Superpixels with SP-InceptionV" + str(args.model_to_use) + "-OnFire"
keepProcessing = True

video = cv2.VideoCapture(args.video_file)
print("Loaded video ...")

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_time = round(1000/fps)

active_state = False

while (keepProcessing):

    start_t = cv2.getTickCount()

    ret, frame = video.read()
    if not ret:
        print("... end of video file reached")
        break

    small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

    slic = cv2.ximgproc.createSuperpixelSLIC(small_frame, region_size=22)
    slic.iterate(10)

    segments = slic.getLabels()

    segment_outputs = []

    for (i, segVal) in enumerate(np.unique(segments)):

        mask = np.zeros(small_frame.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255

        if (int(cv2.__version__.split(".")[0]) >= 4):
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)

        if ((args.model_to_use == 3) or (args.model_to_use == 4)):

            superpixel = cv2.cvtColor(superpixel, cv2.COLOR_BGR2RGB)

            superpixel = pad_image(extract_bounded_nonzero(superpixel), 224, 224)

        output = model.predict([superpixel])

        if round(output[0][0]) == 1: # equiv. to 0.5 threshold in [Dunnings / Breckon, 2018]
            cv2.drawContours(small_frame, contours, -1, (0,255,0), 1)
            segment_outputs.append(True)

        else:
            cv2.drawContours(small_frame, contours, -1, (0,0,255), 1)
            segment_outputs.append(False)

    stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000


    cv2.imshow(windowName, small_frame)

    key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF
    if (key == ord('x')):
        keepProcessing = False
    elif (key == ord('f')):
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Return results
    print("Fire Detected: ", any(segment_outputs))

    centroid_x, centroid_y, pixel_counter = 0, 0, 0
    for x in range(len(segments)):
        for y in range(len(segments[0])):
            if segments[x][y]-1 < len(segment_outputs) and segment_outputs[segments[x][y]-1]:
                centroid_x += x
                centroid_y += y
                pixel_counter += 1
    print("Fire Image Coordinate Centroid: (%d, %d)" % (int(centroid_x / pixel_counter), int(centroid_y / pixel_counter)))


################################################################################

