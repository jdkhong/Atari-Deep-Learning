
import tensorflow as tf
import numpy as np
import gym
from scipy.misc import imresize


class Preprocess(object):

    def get_preprocessed_frame(image, new_HW):
        """Returns a resize image
        Args:
            image (3-D Array): RGB Image Array of shape (H, W, C)
            new_HW (tuple, optional): New Height and Width (height, width)
        Returns:
            3-D Array: A resized grayscale image of shape (`height`, `width`, C)
        """

        return imresize(image, new_HW, interp='nearest')

    def crop_ROI(image, height_range=(35, 195), width_range=(0, 160)):
        """Crops a region of interest (ROI)
        Args:
            image (3-D Array): RGB Image of shape (H, W, C)
            height_range (tuple, optional): Height range to keep (h_begin, h_end)
            width_range (tuple, optional): Width range to keep (w_begin, w_end)
        Returns:
            3-D array: Cropped image of shape (h_end - h_begin, w_end - w_begin, C)
        """
        h_beg, h_end = height_range
        w_beg, w_end = width_range
        return image[h_beg:h_end, w_beg:w_end, ...]

    def kill_background_grayscale(image, bg):
        """Make the background 0
        Args:
            image (3-D array): Numpy array (H, W, C)
            bg (tuple): RGB code of background (R, G, B)
        Returns:
            image (2-D array): Binarized image of shape (H, W)
                The background is 0 and everything else is 1
        """
        R = image[..., 0]
        R[R == 144] = 0
        R[R == 109] = 0
        R[R != 0] = 1

        return R

    def preprocess(image, new_HW):
        """Image process pipeline
        Args:
            image (3-D Array): 3-D array of shape (H, W, C)
            new_HW (tuple): New height and width int tuple of (height, width)
        Returns:
            3-D Array: Binarized image of shape (height, width, 1)
        """
        image = Preprocess.crop_ROI(image)
        image = Preprocess.get_preprocessed_frame(image, new_HW=new_HW)
        image = Preprocess.kill_background_grayscale(image, (144, 72, 17))
        image = np.expand_dims(image, axis=2)
        return image
