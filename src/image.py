import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import random


class Image:
    """
       Class for loading and processing images, providing interactive functionality.

       Attributes:
           image_path (str): Path to the image file.
           img (numpy.ndarray): Loaded image data.
           gray_img (numpy.ndarray): Grayscale version of the image.
           rect_points (list): List to store the points defining the rectangle.
           rectangle_created (bool): Flag to indicate whether a rectangle has been created.
       """
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv.imread(image_path)
        self.gray_img = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        self.rect_points = []
        self.rectangle_created = False

    def display(self):
        """
          Plot the image and enable interactive features.
        """
        plt.imshow(self.gray_img, cmap='gray')
        plt.axis('off')  # Turn off axis
        plt.show()

