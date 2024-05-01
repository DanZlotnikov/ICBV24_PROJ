import cv2
import numpy as np
from src.image import Image
import imageio
import os

from src.image_completion.axis_fft_2d import AxisFFTPartial
from src.image_completion.full_fft_2d import FullFFT2D
from src.image_completion.tile_fft_2d import TileFFT2D

uploads_directory = '../uploads'
processed_directory = '../processed'


def remove_rectangle(image_name, x, delta_x, y, delta_y):
    full_path = uploads_directory + image_name
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

    os.makedirs(uploads_directory, exist_ok=True)
    imageio.imwrite(full_path, modified_image)

    return modified_image


def fft_complete(method, image_name, x, y, delta_x, delta_y):
    full_path = uploads_directory + image_name
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    reconstructed_part_from_full = None

    if method == 'Single':
        model = FullFFT2D().fit(image)
        reconstructed_part_from_full = model.predict(x, y, delta_x, delta_y)
    elif method == 'Full':
        model = AxisFFTPartial.fit(image)
        reconstructed_part_from_full = model.predict(x, y, delta_x, delta_y)
    elif method == 'Tiling':
        model = TileFFT2D.fit(image)
        reconstructed_part_from_full = model.predict(x, y, delta_x, delta_y)

    completed_image = np.copy(image)
    completed_image[y:y+delta_y, x:x+delta_x] = reconstructed_part_from_full

    new_path = os.path.join(processed_directory, image_name)
    imageio.imwrite(new_path, completed_image)
    return True

