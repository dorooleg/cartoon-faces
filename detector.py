#!/usr/bin/env python3
from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np

import effects.effect as effect
import effects.face as face
import effects.mask as mask

masks = {}


def create_effect_pipeline(face_detector, mask_name):
    global masks
    masks, names = mask.Loader(detector=face_detector).load()
    mermaid = masks[mask_name]
    mask_imposter = mask.PlainImposter(mermaid[0], mermaid[1], face_detector)
    pipeline = effect.Pipeline()
    pipeline.add_list([mask_imposter])
    return pipeline, names


def replace_mask(mask_arg, face_detector):
    mask_imposter = mask.PlainImposter(mask_arg[0], mask_arg[1], face_detector)
    pipeline = effect.Pipeline()
    pipeline.add_list([mask_imposter])
    return pipeline


def replace_faces(faces_name, face_detector):

    print("replace name on" + faces_name)
    return replace_mask(masks[faces_name], face_detector)


def init(mask_name):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print('[INFO] loading facial landmark predictor...')
    face_detector = face.Detector()
    pipeline, image_names = create_effect_pipeline(face_detector, mask_name)

    # initialize the video stream and allow the cammera sensor to warmup
    print('[INFO] camera sensor warming up...')
    vs = VideoStream().start()
    time.sleep(2.0)
    return vs, pipeline, face_detector, image_names


def create(vs, pipeline):
    im1 = vs.read()
    im1 = imutils.resize(im1, width=400)
    if hasattr(pipeline, '__len__') and len(pipeline) > 1:
        pipeline = pipeline[0]
    res = pipeline.process(im1)
    return res


def destroy(vs):
    cv2.destroyAllWindows()
    vs.stop()
