#!/usr/bin/env python3
from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np
import sys

import effects.face as face
import effects.mask as mask

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print('[INFO] loading facial landmark predictor...')
face_detector = face.Detector()
masks = mask.Loader(detector=face_detector).load()
mask_imposter = mask.PlainImposter(masks['mermaid'][0], masks['mermaid'][1])


# initialize the video stream and allow the cammera sensor to warmup
print('[INFO] camera sensor warming up...')
vs = VideoStream().start()
time.sleep(2.0)


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
