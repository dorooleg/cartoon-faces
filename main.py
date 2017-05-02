#!/usr/bin/env python3
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import sys

import effects.face as face

xs = [142, 138, 131, 128, 130, 144, 157, 180, 203, 226, 255, 286, 313, 325, 332, 339, 345, 153, 177, 195, 217, 231, 264, 294, 315, 338, 350, 241, 238, 237, 227, 202, 210, 221, 229, 241, 160, 184, 211, 219, 194, 169, 266, 287, 314, 333, 318, 290, 163, 182, 203, 218, 237, 254, 271, 255, 237, 208, 187, 177, 169, 194, 217, 231, 249, 232, 217, 196]
ys = [201, 222, 248, 281, 304, 346, 356, 381, 391, 390, 375, 353, 331, 315, 293, 267, 250, 170, 164, 169, 182, 200, 209, 201, 199, 201, 215, 224, 247, 267, 288, 297, 303, 307, 309, 306, 205, 194, 207, 233, 236, 230, 249, 228, 226, 245, 265, 263, 306, 311, 316, 321, 322, 325, 329, 346, 359, 358, 346, 334, 321, 322, 343, 242, 339, 336, 337, 332]
pts2 = np.zeros((len(xs), 2), dtype=int)
pts2[:, 0] = np.array(xs)
pts2[:, 1] = np.array(ys)


mermaid_img = cv2.imread('./data/images/mermaid.png', cv2.IMREAD_UNCHANGED)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR_FRAC = 0.6

# ========================================================

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in face.OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    st = np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T))
    nl = np.matrix([0., 0., 1.])
    print(st.shape, nl.shape)
    st = st[:, 0:3]

    return np.vstack([st, nl])


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

# ========================================================

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print('[INFO] loading facial landmark predictor...')
face_detector = face.Detector()

# initialize the video stream and allow the cammera sensor to warmup
print('[INFO] camera sensor warming up...')
vs = VideoStream().start()
time.sleep(2.0)


while True:

    im1 = vs.read()
    im1 = imutils.resize(im1, width=400)
    im1 = cv2.resize(im1, (im1.shape[1], im1.shape[0]))
    landmarks1 = face_detector.detect(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2BGRA)

    im2 = mermaid_img
    im2 = cv2.resize(im2, (im2.shape[1], im2.shape[0]))
    landmarks2 = pts2[:]

    output_im = None

    if len(landmarks1) >= 1:
        landmarks1 = landmarks1[0]
        M = transformation_from_points(landmarks1[face.ALIGN_POINTS],
                                       landmarks2[face.ALIGN_POINTS])

        mask = get_face_mask(im2, landmarks2)
        warped_im2 = warp_im(im2, M, im1.shape)
        alpha = (warped_im2[:, :, 0] == 0) * 255
        beta = 255 - alpha

        for i in range(3):
            im1[:, :, i] = np.bitwise_and(im1[:, :, i], alpha)
            warped_im2[:, :, i] = np.bitwise_and(warped_im2[:, :, i], beta)

        output_im = warped_im2 + im1
    else:
        output_im = im1

    cv2.imshow('Frame', output_im)
    # cv2.imshow('Frame', memaid_img)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()