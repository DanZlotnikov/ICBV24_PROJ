from src.image import Image
import matplotlib.pyplot as plt
from src.image_completion import *
import cv2
from src.image_completion.full_fft_2d import FullFFT2D
from src.image_completion.axis_fft_2d import AxisFFT2D

if __name__ == '__main__':

    # Example 1: Image completion using full FFT 2d
    lenna_img = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)
    model = FullFFT2D().fit(lenna_img)
    corrupted_x, corrupted_y, corrupted_width, corrupted_height = 100, 100, 20, 20
    reconstructed_part_from_full = model.predict(corrupted_x, corrupted_y, corrupted_width, corrupted_height)
    
    # Example 2: Image completion using axis FFT 2d
    model = AxisFFT2D().fit(lenna_img)
    reconstructed_part_from_axis = model.predict(corrupted_x, corrupted_y, corrupted_width, corrupted_height)
    
    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(lenna_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')  # Turn off axis
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_part_from_full, cmap='gray')
    plt.title('Reconstructed Image (Full FFT 2D)')
    plt.axis('off')  # Turn off axis
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_part_from_axis, cmap='gray')
    plt.title('Reconstructed Image (Axis FFT 2D)')
    plt.axis('off')  # Turn off axis
    plt.show()
    
    modified = remove_rectangle('./nadal.jpg', 100, 100, 500, 500)
    plt.imshow(modified, cmap='gray')
    plt.axis('off')  # Turn off axis
    plt.show()


