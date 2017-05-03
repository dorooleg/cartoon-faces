#!/usr/bin/env python3
import os
import cv2
import numpy as np
import effects.effect as effect
import effects.face as face
import random

def get_transposition(pts1, pts2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    pts1 = pts1.astype(np.float64)
    pts2 = pts2.astype(np.float64)

    c1 = np.mean(pts1, axis=0)
    c2 = np.mean(pts2, axis=0)
    pts1 -= c1
    pts2 -= c2

    s1 = np.std(pts1)
    s2 = np.std(pts2)
    pts1 /= s1
    pts2 /= s2

    U, S, Vt = np.linalg.svd(pts1.T * pts2)
    R = (U * Vt).T

    st = np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T))[:, 0:3]
    nl = np.matrix([0., 0., 1.])
    return np.vstack([st, nl])


def _traverse_dir(path):
    for filename in os.listdir(path):
        name = os.path.splitext(filename)[0]
        filepath = os.path.join(path, filename)
        if name.startswith('.'):
            continue
        yield (name, filepath)


class Loader:
    def __init__(self, **kargs):
        if 'detector' in kargs:
            self.detector = kargs['detector']
        else:
            self.detector = face.Detector()

    def load(self):
        images_path = './data/images'
        marks_path = './data/mask_landmarks'
        images, marks = {}, {}

        list_name_mask = {}
        for name, path in _traverse_dir(marks_path):
            data = np.genfromtxt(path, delimiter=',', loose=True, invalid_raise=False)
            if data is not None and data.size > 0:
                marks[name] = np.transpose(data)

        for name, path in _traverse_dir(images_path):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:
                if name in marks:
                    images[name] = image, marks[name]
                    continue
                faces = self.detector.detect(image)
                if len(faces) > 0:
                    images[name] = image, faces[0]
            list_name_mask[name] = path
        return images, list_name_mask


class PlainImposter(effect.Effect):
    def __init__(self, mask_image, mask_markup, detector):
        self.image = mask_image
        self.markup = mask_markup
        self.markup_align = mask_markup[face.ALIGN_POINTS]
        self.detector = detector
        # self.name_all_mask = name_all_mask
        # move = [[0, 0, -10],
        #         [0, 0, -125], # -50
        #         [0, 0, 0]]
        move = [[0, 0, 0],
                [0, 0, 30],  # -50
                [0, 0, 0]]
        # move = [[0., 0, -10],
        #         [0, 0.1, 0], # -50
        #         [0, 0, 0]]
        self.move = np.array(move)

    def __warp_im(self, transposition, dshape):
        res = np.zeros(dshape, dtype=self.image.dtype)
        transposition += self.move
        cv2.warpAffine(self.image,
                       transposition[:2],
                       (dshape[1], dshape[0]),
                       dst=res,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return res

    def process(self, image):
        res = image[:]
        for marks in self.detector.detect(image):
            M = get_transposition(marks[face.ALIGN_POINTS], self.markup_align)
            warped_im2 = self.__warp_im(M, res.shape)
            alpha = (warped_im2 == 0) * 255
            beta = 255 - alpha

            res[:] = np.bitwise_and(res, alpha)
            warped_im2[:] = np.bitwise_and(warped_im2[:], beta)
            res += warped_im2
        return res
