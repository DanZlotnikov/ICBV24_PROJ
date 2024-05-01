import numpy as np


def remove_rectangle(image, x, y, delta_x, delta_y):
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
    return modified_image
