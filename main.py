#!/usr/bin/env python3
from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np
import sys

import effects.face as face
import effects.mask as mask

xs = [142, 138, 131, 128, 130, 144, 157, 180, 203, 226, 255, 286, 313, 325, 332, 339, 345, 153, 177, 195, 217, 231, 264, 294, 315, 338, 350, 241, 238, 237, 227, 202, 210, 221, 229, 241, 160, 184, 211, 219, 194, 169, 266, 287, 314, 333, 318, 290, 163, 182, 203, 218, 237, 254, 271, 255, 237, 208, 187, 177, 169, 194, 217, 231, 249, 232, 217, 196]
ys = [201, 222, 248, 281, 304, 346, 356, 381, 391, 390, 375, 353, 331, 315, 293, 267, 250, 170, 164, 169, 182, 200, 209, 201, 199, 201, 215, 224, 247, 267, 288, 297, 303, 307, 309, 306, 205, 194, 207, 233, 236, 230, 249, 228, 226, 245, 265, 263, 306, 311, 316, 321, 322, 325, 329, 346, 359, 358, 346, 334, 321, 322, 343, 242, 339, 336, 337, 332]
pts2 = np.zeros((len(xs), 2), dtype=int)
pts2[:, 0] = np.array(xs)
pts2[:, 1] = np.array(ys)

mermaid_img = cv2.imread('./data/images/mermaid.png', cv2.IMREAD_COLOR)


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print('[INFO] loading facial landmark predictor...')
face_detector = face.Detector()

# initialize the video stream and allow the cammera sensor to warmup
print('[INFO] camera sensor warming up...')
vs = VideoStream().start()
time.sleep(2.0)

mask_imposter = mask.PlainImposter(mermaid_img, pts2)


while True:
    im1 = vs.read()
    im1 = imutils.resize(im1, width=400)
    landmarks1 = face_detector.detect(im1)

    res = mask_imposter.impose(im1, landmarks1)
    cv2.imshow('Frame', res)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


cv2.destroyAllWindows()
vs.stop()
