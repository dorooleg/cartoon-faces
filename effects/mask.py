#!/usr/bin/env python3
import cv2
import numpy as np
import effects.face as face


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


class PlainImposter:
    def __init__(self, mask_image, mask_markup):
        self.image = mask_image
        self.markup = mask_markup
        self.markup_align = mask_markup[face.ALIGN_POINTS]

    def __warp_im(self, transposition, dshape):
        res = np.zeros(dshape, dtype=self.image.dtype)
        cv2.warpAffine(self.image,
                       transposition[:2],
                       (dshape[1], dshape[0]),
                       dst=res,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return res

    def impose(self, image, landmarks1):
        res = image[:]

        for marks in landmarks1:
            M = get_transposition(marks[face.ALIGN_POINTS], self.markup_align)
            warped_im2 = self.__warp_im(M, res.shape)
            alpha = (warped_im2 == 0) * 255
            beta = 255 - alpha

            res[:] = np.bitwise_and(res, alpha)
            warped_im2[:] = np.bitwise_and(warped_im2[:], beta)
            res += warped_im2

        return res
