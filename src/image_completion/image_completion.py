import cv2
import numpy as np
from src.image import Image
import imageio
import os

def remove_rectangle(image_path, x, delta_x, y, delta_y):
    full_path = "../uploads/"+image_path
    image = Image(full_path)
    height, width = image.gray_img.shape

    # Ensure the rectangle coordinates are within bounds
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    delta_x = max(0, min(delta_x, width - x))
    delta_y = max(0, min(delta_y, height - y))

    # Create a copy of the image
    modified_image = np.copy(image.gray_img)

    # Remove the rectangle from the image
    modified_image[y:delta_y+y, x:delta_x+x] = 0


    output_directory = '../uploads'
    os.makedirs(output_directory, exist_ok=True)
    imageio.imwrite(os.path.join(output_directory, image_path), modified_image)

    return modified_image


def fft_complete_tiles(image, x, y, delta_x, delta_y):
    updated_image = np.copy(image)
    num_tiles = 5
    diff_x = int(delta_x / num_tiles)
    diff_y = int(delta_y / num_tiles)

    for i in range(num_tiles):
        for j in range(num_tiles):
            start_x = x + i * diff_x
            end_x = x + (i + 1) * diff_x
            start_y = y + j * diff_y
            end_y = y + (j + 1) * diff_y
            reconstructed_part = fft_single_tile(updated_image, start_x, start_y, diff_x, diff_y)
            updated_image[start_x: end_x, start_y: end_y] = reconstructed_part

    return updated_image


def fft_single_tile(image, x, y, delta_x, delta_y):
    model = FullFFT2D().fit(image)
    return model.predict(x, y, delta_x, delta_y)
