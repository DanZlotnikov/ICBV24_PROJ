from src.image import Image
import matplotlib.pyplot as plt
from src.image_completion import *
import cv2
from src.image_completion.full_fft_2d import FullFFT2D

if __name__ == '__main__':

    # Example 1: Image completion using full FFT 2d
    lenna_img = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
    model = FullFFT2D().fit(lenna_img)
    corrupted_x, corrupted_y, corrupted_width, corrupted_height = 100, 100, 20, 20
    reconstructed_part = model.predict(corrupted_x, corrupted_y, corrupted_width, corrupted_height)
    
    
    modified = remove_rectangle('./nadal.jpg', 100, 100, 500, 500)
    plt.imshow(modified, cmap='gray')
    plt.axis('off')  # Turn off axis
    plt.show()


