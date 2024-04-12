import numpy as np
import cv2


class Difference_of_Gaussian(object):

    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1 / 4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def findkeypoints(self, images: np.ndarray, octave: int) -> set:
        keypoints = set()
        row, column = images[0].shape
        for x in range(1, row - 2):
            for y in range(1, column - 2):
                for DoG in range(1, self.num_DoG_images_per_octave - 1):
                    pixel = images[DoG, x, y]
                    cube = images[DoG - 1:DoG + 2, x - 1:x + 2,
                                  y - 1:y + 2].flatten()
                    if (np.absolute(pixel) >= self.threshold
                            and (pixel >= max(cube) or pixel <= min(cube))):
                        keypoints.add((x * octave, y * octave))
        return keypoints

    def save_dog_images(self, images: np.ndarray) -> None:
        for idx, img in enumerate(images):
            max_val = max(img.flatten())
            min_val = min(img.flatten())
            norm_img = (img - min_val) * 255 / (max_val - min_val)
            cv2.imwrite(f'./result/DoG_1_{idx+1}.png', norm_img)

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []

        # 1st octave
        first_octave = [image]
        first_octave.extend(
            cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=self.sigma**idx)
            for idx in range(1, self.num_guassian_images_per_octave))
        # 2nd octave
        resize_factor = 0.5
        second_octave = [
            cv2.resize(
                first_octave[-1],
                None,  # dsize
                fx=resize_factor,
                fy=resize_factor,
                interpolation=cv2.INTER_NEAREST)  # INTER_LENEAR is default
        ]
        second_octave.extend(
            cv2.GaussianBlur(
                second_octave[0], ksize=(0, 0), sigmaX=self.sigma**idx)
            for idx in range(1, self.num_guassian_images_per_octave))

        # gaussian_images.extend(first_octave)
        # gaussian_images.extend(second_octave)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []

        dog_images = [0] * (self.num_DoG_images_per_octave * self.num_octaves)
        for idx in range(self.num_DoG_images_per_octave):
            dog_images[idx] = cv2.subtract(first_octave[idx],
                                           first_octave[idx + 1])
            dog_images[idx + self.num_DoG_images_per_octave] = cv2.subtract(
                second_octave[idx], second_octave[idx + 1])

        # save images
        # self.save_dog_images(dog_images[4:])

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        keypoints = set()
        keypoints.update(
            self.findkeypoints(
                np.array(dog_images[0:self.num_DoG_images_per_octave]), 1))
        keypoints.update(
            self.findkeypoints(
                np.array(dog_images[self.num_DoG_images_per_octave:]), 2))
        keypoints = list(map(list, keypoints))

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique

        keypoints = np.unique(np.array(keypoints), axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        return keypoints
