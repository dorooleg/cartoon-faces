#!/usr/bin/env python3
import dlib
import numpy as np


JAW_POINTS = list(range(0, 17))
FACE_POINTS = list(range(17, 68))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 35))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 61))

ALIGN_POINTS = LEFT_BROW_POINTS + RIGHT_EYE_POINTS + \
                LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]


class Detector:
    PREDICTOR = './data/classifiers/shape_predictor_68_face_landmarks.dat'

    def __init__(self, **kargs):
        predictor_path = Detector.PREDICTOR
        if 'predictor_path' in kargs:
            predictor_path = kargs[predictor_path]

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect(self, image):
        rects = self.detector(image, 1)

        faces = []
        for rect in rects:
            parts = self.predictor(image, rect).parts()
            points = np.matrix([[p.x, p.y] for p in parts])
            faces.append(points)
        return faces
