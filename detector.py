#!/usr/bin/env python3
from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np

import effects.effect as effect
import effects.face as face
import effects.mask as mask


def create_effect_pipeline(detector, face_detector):
    masks = mask.Loader(detector=face_detector).load()
    mermaid = masks['mermaid']
    mask_imposter = mask.PlainImposter(mermaid[0], mermaid[1], face_detector)

    pipeline = effect.Pipeline()
    pipeline.add_list([mask_imposter])

    return pipeline


def init():
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print('[INFO] loading facial landmark predictor...')
    face_detector = face.Detector()
    pipeline = create_effect_pipeline(face_detector, face_detector)

    # initialize the video stream and allow the cammera sensor to warmup
    print('[INFO] camera sensor warming up...')
    vs = VideoStream().start()
    time.sleep(2.0)
    return vs, pipeline


def create(vs, pipeline):
    im1 = vs.read()
    im1 = imutils.resize(im1, width=400)
    res = pipeline.process(im1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        return
    print("create res")
    return res


def destroy(vs):
    cv2.destroyAllWindows()
    vs.stop()
