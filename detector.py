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
masks_move = {}
mask_imposter = None


def create_effect_pipeline(mask_name):
    global masks_move, masks, mask_imposter
    masks, names, masks_move = mask.Loader().load()
    mermaid = masks[mask_name]
    move = masks_move[mask_name]
    mask_imposter = mask.PlainImposter(mermaid[0], mermaid[1], masks, move)
    pipeline = effect.Pipeline()
    pipeline.add_list([mask_imposter])
    return pipeline, names

def replace_mask(mask_arg, mask_move):
    global mask_imposter
    # print(mask_imposter)
    mask_imposter = mask_imposter.set_mask(mask_arg[0], mask_arg[1], mask_move)
    pipeline = effect.Pipeline()
    pipeline.add_list([mask_imposter])
    return pipeline

def replace_faces(faces_name):
    print("replace name on " + faces_name)
    return replace_mask(masks[faces_name], masks_move[faces_name])


def init(mask_name):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print('[INFO] loading facial landmark predictor...')
    pipeline, image_names = create_effect_pipeline(mask_name)

    # initialize the video stream and allow the cammera sensor to warmup
    print('[INFO] camera sensor warming up...')
    vs = VideoStream().start()
    time.sleep(2.0)
    return vs, pipeline, image_names


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
