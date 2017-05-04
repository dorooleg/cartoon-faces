#!/usr/bin/env python3
from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np

import effects.effect as effect
import effects.face as face
import effects.mask as mask

def create_effect_pipeline(mask_name, masks):
    cur_mask = masks[mask_name]
    mask_imposter = mask.PlainImposter(cur_mask[0], cur_mask[1], masks, cur_mask[2])
    pipeline = effect.Pipeline()
    pipeline.add_list([mask_imposter])
    return pipeline


def get_video_stream():
    # initialize the video stream and allow the cammera sensor to warmup
    print('[INFO] camera sensor warming up...')
    vs = VideoStream().start()
    time.sleep(2.0)
    return vs


def create(vs, pipeline):
    im1 = vs.read()
    im1 = imutils.resize(im1, width=400)
    res = pipeline.process(im1)
    return res


def destroy(vs):
    cv2.destroyAllWindows()
    vs.stop()
