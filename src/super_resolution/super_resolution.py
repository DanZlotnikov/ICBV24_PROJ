import numpy as np
from scipy.ndimage import gaussian_filter

def dist(x,y,thres):
    d= np.linalg.norm(x-y)
    if d > thres:
        return 0.9
    else:
        return 0.01

def upscale(image, scale_factor,thres):
    """
    upscales the image by the scale factor using interpoltion to assign the new values to the upscaled image(initial labels
    :param image:
    :param scale_factor:
    :return:
    """
    height, width, channels = image.shape

    # Create an upscaled image with the same dimensions
    upscaled_image = np.zeros_like(image, dtype=np.uint8)
    initial_confidence = {}
    base_pixels = {}
    # Perform relaxation labeling
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                # Interpolate additional pixels between the original pixels
                for dy in range(scale_factor):
                    for dx in range(scale_factor):
                        if dx == 0 and dy == 0:
                            upscaled_image[y * scale_factor + dy, x * scale_factor + dx, c] = image[y, x, c]
                            base_pixels[(y, x, c)] = image[y, x, c]
                        else:
                            weight_x = dx / scale_factor
                            weight_y = dy / scale_factor

                            upscaled_image[y * scale_factor + dy, x * scale_factor + dx, c] = (
                                    (1 - weight_x) * (1 - weight_y) * image[y, x, c] +
                                    weight_x * (1 - weight_y) * image[y, min(x + 1, width - 1), c] +
                                    (1 - weight_x) * weight_y * image[min(y + 1, height - 1), x, c] +
                                    weight_x * weight_y * image[min(y + 1, height - 1), min(x + 1, width - 1), c]
                            )
                        initial_confidence[y * scale_factor + dy, x * scale_factor + dx, c] = dist(base_pixels[(y, x, c)],
                                                                                                   upscaled_image[y * scale_factor + dy, x * scale_factor + dx, c],thres)

    return upscaled_image, base_pixels, initial_confidence


def gaussian_kernel(image_shape, sigma):
    """
    Generate a 2D Gaussian kernel based on the size of the input image and the specified standard deviation (sigma).
    """
    # Calculate the size of the kernel based on the sigma value
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)

    # Generate the 2D Gaussian kernel using scipy's gaussian_filter function
    kernel = gaussian_filter(np.zeros(image_shape), sigma=sigma)

    return kernel / np.sum(kernel)


def relaxation_labeling(image,scale_factor):
    upscaled_image,base_pixels, initial_confidence = upscale(image, scale_factor, 0.5)
    # ____
    kernel = gaussian_kernel(upscaled_image.shape, scale_factor)

    pass


if __name__ == '__main__':
    print('hello')

