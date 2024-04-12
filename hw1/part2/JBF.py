import numpy as np
import cv2
from tqdm import trange


class Joint_bilateral_filter(object):

    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w,
                                        self.pad_w, self.pad_w,
                                        BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w,
                                             self.pad_w, self.pad_w,
                                             BORDER_TYPE).astype(np.int32)
        ### TODO ###
        # G_s(p, q) = e ^ (-(((x_p - x_q)^2 + (y_p - y_q)^2) / (2 * sigma_s ^ 2)))
        table_G_s = np.exp(-(np.arange(self.pad_w + 1)**2) /
                           (2 * self.sigma_s**2))

        # T is single-channel:  G_r(T_p, T_q) = e ^ (-((T_p-T_q)^2 / (2 * sigma_s ^ 2)))
        # T is color image:     G_r(T_p, T_q) = e ^ (-(((T_p^r-T_q^r)^2+(T_p^q-T_q^q)^2+(T_p^b-T_q^b)^2) / (2 * sigma_s ^ 2)))
        table_G_r = np.exp(-((np.arange(256) / 255)**2 /
                             (2 * self.sigma_r**2)))

        hwc = padded_img.shape
        weight = np.zeros(hwc)
        result = np.zeros(hwc)

        for x in trange(-self.pad_w, self.pad_w + 1):
            for y in range(-self.pad_w, self.pad_w + 1):
                dT = table_G_r[np.abs(
                    np.roll(padded_guidance, [y, x], axis=[0, 1]) -
                    padded_guidance)]
                r_weight = dT if dT.ndim == 2 else np.prod(dT, axis=2)
                s_weight = table_G_s[np.abs(x)] * table_G_s[np.abs(y)]
                t_weight = s_weight * r_weight
                padded_img_roll = np.roll(padded_img, [y, x], axis=[0, 1])
                for channel in range(padded_img.ndim):
                    result[:, :,
                           channel] += padded_img_roll[:, :,
                                                       channel] * t_weight
                    weight[:, :, channel] += t_weight
        output = (result / weight)[self.pad_w:-self.pad_w,
                                   self.pad_w:-self.pad_w, :]

        return np.clip(output, 0, 255).astype(np.uint8)
