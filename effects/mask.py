#!/usr/bin/env python3
import os
import cv2
import numpy as np
import effects.effect as effect
import effects.face as face
import random

import effects.face as face


class DetectorSingleton:
    face_detector = None

    def get():
        if DetectorSingleton.face_detector is None:
            DetectorSingleton.face_detector = face.Detector()
        return DetectorSingleton.face_detector


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
    def __init__(self):
        self.detector = DetectorSingleton.get()

    def load(self):
        images_path = './data/images'
        marks_path = './data/mask_landmarks'
        move_path = './data/mask_move'
        images, marks, move = {}, {}, {}
        list_name_mask = {}
        for name, path in _traverse_dir(marks_path):
            data = np.genfromtxt(path, delimiter=',', loose=True, invalid_raise=False)
            if data is not None and data.size > 0:
                marks[name] = np.transpose(data)

        for name, path in _traverse_dir(move_path):
            data = np.genfromtxt(path, delimiter=',', loose=True, invalid_raise=False)
            if data is not None and data.size > 0:
                move[name] = np.transpose(data)

        for name, path in _traverse_dir(images_path):
            list_name_mask[name] = path
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:
                if name in marks:
                    images[name] = image, marks[name], move[name]
                    continue
                faces = self.detector.detect(image)
                if len(faces) > 0:
                    images[name] = image, faces[0], move[name]

        return images, list_name_mask


class PlainImposter(effect.Effect):
    def __init__(self, mask_image, mask_markup, all_mask, mask_move):
        self.image = mask_image
        self.markup = mask_markup
        self.markup_align = mask_markup[face.ALIGN_POINTS]
        self.detector = DetectorSingleton.get()
        self.all_mask = all_mask
        self.size_faces = 10
        name = [name for name, _ in all_mask.items()]
        self.faces_name = [random.choice(name) for _ in range(self.size_faces)]
        self.images = [all_mask[name][0] for name in self.faces_name]
        self.markups = [all_mask[name][1] for name in self.faces_name]
        self.moves = [all_mask[name][2] for name in self.faces_name]
        self.markup_aligns = [item[face.ALIGN_POINTS] for item in self.markups]
        self.move = mask_move
        move = [[0, 0, self.move[0]],
                [0, 0, self.move[1]],
                [0, 0, 0]]
        self.move = np.array(move)

    def set_mask(self, mask_image, mask_markup, mask_move):
        self.move = mask_move
        move = [[0, 0, self.move[0]],
                [0, 0, self.move[1]],
                [0, 0, 0]]
        self.move = np.array(move)
        self.image = mask_image
        self.markup = mask_markup
        self.markup_align = mask_markup[face.ALIGN_POINTS]
        return self

    def __warp_im(self, transposition, dshape, marks, idx=0):
        res = np.zeros(dshape, dtype=self.image.dtype)
        if idx > 0:
            move = [[0, 0, self.moves[idx - 1][0]],
                    [0, 0, self.moves[idx - 1][1]],
                    [0, 0, 0]]
            move = np.array(move)
            transposition = get_transposition(marks[face.ALIGN_POINTS], self.markup_aligns[idx - 1]) + move
            cv2.warpAffine(self.images[idx - 1],
                           transposition[:2],
                           (dshape[1], dshape[0]),
                           dst=res,
                           borderMode=cv2.BORDER_TRANSPARENT,
                           flags=cv2.WARP_INVERSE_MAP)
        else:
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
        print("face count :", len(self.detector.detect(image)))
        for idx, marks in enumerate(self.detector.detect(image)):
            M = get_transposition(marks[face.ALIGN_POINTS], self.markup_align)
            warped_im2 = self.__warp_im(M, res.shape, marks, idx)
            alpha = (warped_im2 == 0) * 255
            beta = 255 - alpha
            res[:] = np.bitwise_and(res, alpha)
            warped_im2[:] = np.bitwise_and(warped_im2[:], beta)
            res += warped_im2
        return res
