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

xs = [142, 138, 131, 128, 130, 144, 157, 180, 203, 226, 255, 286, 313, 325, 332, 339, 345, 153, 177, 195, 217, 231, 264, 294, 315, 338, 350, 241, 238, 237, 227, 202, 210, 221, 229, 241, 160, 184, 211, 219, 194, 169, 266, 287, 314, 333, 318, 290, 163, 182, 203, 218, 237, 254, 271, 255, 237, 208, 187, 177, 169, 194, 217, 231, 249, 232, 217, 196]
ys = [201, 222, 248, 281, 304, 346, 356, 381, 391, 390, 375, 353, 331, 315, 293, 267, 250, 170, 164, 169, 182, 200, 209, 201, 199, 201, 215, 224, 247, 267, 288, 297, 303, 307, 309, 306, 205, 194, 207, 233, 236, 230, 249, 228, 226, 245, 265, 263, 306, 311, 316, 321, 322, 325, 329, 346, 359, 358, 346, 334, 321, 322, 343, 242, 339, 336, 337, 332]
pts2 = np.zeros((len(xs), 2), dtype=int)
pts2[:, 0] = np.array(xs)
pts2[:, 1] = np.array(ys)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True,
                help='path to facial landmark predictor')
ap.add_argument('-r', '--picamera', type=int, default=-1,
                help='whether or not the Raspberry Pi camera should be used')
args = vars(ap.parse_args())

mermaid_img = cv2.imread('./images/mermaid.png', cv2.IMREAD_UNCHANGED)


SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])


# ========================================================

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) != 1:
        return None

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
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

    print(points1.shape, points2.shape)

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

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))

# ========================================================

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print('[INFO] loading facial landmark predictor...')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

# initialize the video stream and allow the cammera sensor to warmup
print('[INFO] camera sensor warming up...')
vs = VideoStream(usePiCamera=args['picamera'] > 0).start()
time.sleep(2.0)


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    im = im / 255.0

    return im, s


while True:

    im1 = vs.read()
    im1 = imutils.resize(im1, width=400)
    im1 = cv2.resize(im1, (im1.shape[1], im1.shape[0]))
    landmarks1 = get_landmarks(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2BGRA)

    im2 = mermaid_img
    im2 = cv2.resize(im2, (im2.shape[1], im2.shape[0]))
    landmarks2 = pts2[:]

    # print(landmarks1.shape, landmarks2.shape)

    # im1, landmarks1 = im1s, landmarks1s
    # im2, landmarks2 = im2s, landmarks2s

    output_im = None

    if landmarks1 is not None:
        M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                       landmarks2[ALIGN_POINTS])

        mask = get_face_mask(im2, landmarks2)
        # warped_mask = warp_im(mask, M, im1.shape)



        # combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)
        # print(combined_mask)

        warped_im2 = warp_im(im2, M, im1.shape)
        alpha = (warped_im2[:, :, 0] == 0) * 255
        beta = 255 - alpha

        for i in range(3):
            im1[:, :, i] = np.bitwise_and(im1[:, :, i], alpha)
            warped_im2[:, :, i] = np.bitwise_and(warped_im2[:, :, i], beta)

        # warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

        # print(1)
        output_im = warped_im2 + im1
        # output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
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