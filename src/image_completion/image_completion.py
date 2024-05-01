import numpy as np
from src.image import Image
import imageio
import os

def remove_rectangle(image_path, x, y, delta_x, delta_y):
    image = Image(image_path)
    height, width = image.gray_img.shape

    # Ensure the rectangle coordinates are within bounds
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    delta_x = max(0, min(delta_x, width - x))
    delta_y = max(0, min(delta_y, height - y))

    # Create a copy of the image
    modified_image = np.copy(image.gray_img)

    # Remove the rectangle from the image
    modified_image[y:y + delta_y, x:x + delta_x] = 0

    output_directory = './uploads'
    os.makedirs(output_directory, exist_ok=True)
    imageio.imwrite(os.path.join(output_directory, image_path), modified_image)

    return modified_image


def fft_complete_tiles(image, x, y, delta_x, delta_y):
    pass