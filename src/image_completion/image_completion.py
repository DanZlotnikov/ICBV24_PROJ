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
