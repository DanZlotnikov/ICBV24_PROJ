# import numpy as np
# from scipy.ndimage import gaussian_filter
# import cv2
#
# def dist(x,y,thres):
#     d= np.linalg.norm(x-y)
#     if d > thres:
#         return 0.9
#     else:
#         return 0.01
#
# def upscale(image, scale_factor,thres):
#     """
#     upscales the image by the scale factor using interpoltion to assign the new values to the upscaled image(initial labels
#     :param image:
#     :param scale_factor:
#     :return:
#     """
#     height, width, channels = image.shape
#
#     # Create an upscaled image with the same dimensions
#     upscaled_image = np.zeros_like(image, dtype=np.uint8)
#     initial_confidence = {}
#     base_pixels = {}
#     # Perform relaxation labeling
#     for y in range(height):
#         for x in range(width):
#             for c in range(channels):
#                 # Interpolate additional pixels between the original pixels
#                 for dy in range(scale_factor):
#                     for dx in range(scale_factor):
#                         if dx == 0 and dy == 0:
#                             upscaled_image[y * scale_factor + dy, x * scale_factor + dx, c] = image[y, x, c]
#                             base_pixels[(y, x, c)] = image[y, x, c]
#                         else:
#                             weight_x = dx / scale_factor
#                             weight_y = dy / scale_factor
#
#                             upscaled_image[y * scale_factor + dy, x * scale_factor + dx, c] = (
#                                     (1 - weight_x) * (1 - weight_y) * image[y, x, c] +
#                                     weight_x * (1 - weight_y) * image[y, min(x + 1, width - 1), c] +
#                                     (1 - weight_x) * weight_y * image[min(y + 1, height - 1), x, c] +
#                                     weight_x * weight_y * image[min(y + 1, height - 1), min(x + 1, width - 1), c]
#                             )
#                         initial_confidence[y * scale_factor + dy, x * scale_factor + dx, c] = dist(base_pixels[(y, x, c)],
#                                                                                                    upscaled_image[y * scale_factor + dy, x * scale_factor + dx, c],thres)
#
#     return upscaled_image, base_pixels, initial_confidence
#
#
# def gaussian_kernel(image_shape, sigma):
#     """
#     Generate a 2D Gaussian kernel based on the size of the input image and the specified standard deviation (sigma).
#     """
#     # Calculate the size of the kernel based on the sigma value
#     kernel_size = int(2 * np.ceil(2 * sigma) + 1)
#
#     # Generate the 2D Gaussian kernel using scipy's gaussian_filter function
#     kernel = gaussian_filter(np.zeros(image_shape), sigma=sigma)
#
#     return kernel / np.sum(kernel)
#
#
# def relaxation_labeling(image,scale_factor):
#     upscaled_image,base_pixels, initial_confidence = upscale(image, scale_factor, 0.5)
#     # ____
#     kernel = gaussian_kernel(upscaled_image.shape, scale_factor)
#
#     return upscaled_image, base_pixels, initial_confidence
#
#
# if __name__ == '__main__':
#     # img =
#     img = cv2.imread("/Users/brahan/PycharmProjects/ICBV24_PROJ/src/nadal.jpg", cv2.IMREAD_GRAYSCALE)
#     print(relaxation_labeling(img,0.5))
# # # See PyCharm help at https://www.jetbrains.com/help/pycharm/
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2


def dist(x,y,thres): # calc the distance between pixel values and set the confidence accordingly
    # need to set it in a gaussian way
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
    height, width = image.shape

    # Create an upscaled image with the same dimensions
    n_img = np.resize(image,( height*scale_factor,width*scale_factor))
    n_image = np.zeros_like(n_img, dtype=np.uint8)
    upscaled_image = np.zeros_like(n_img, dtype=np.uint8)

    initial_confidence = np.zeros_like(n_img, dtype=np.float32)
    base_pixels = {}
    for y in range(height):
        for x in range(width):
            for dy in range(scale_factor):
                for dx in range(scale_factor):
                    if dx == 0 and dy == 0:
                        upscaled_image[y * scale_factor + dy, x * scale_factor + dx] = image[y, x]
                        base_pixels[(y, x)] = image[y, x]
                    else:
                        weight_x = dx / scale_factor
                        weight_y = dy / scale_factor

                        upscaled_image[y * scale_factor + dy, x * scale_factor + dx] = (
                                (1 - weight_x) * (1 - weight_y) * image[y, x] +
                                weight_x * (1 - weight_y) * image[y, min(x + 1, width - 1)] +
                                (1 - weight_x) * weight_y * image[min(y + 1, height - 1), x] +
                                weight_x * weight_y * image[min(y + 1, height - 1), min(x + 1, width - 1)]
                        )
                    initial_confidence[y * scale_factor + dy, x * scale_factor + dx] = dist(base_pixels[(y, x)], upscaled_image[y * scale_factor + dy, x * scale_factor + dx], thres)

    return upscaled_image, base_pixels, initial_confidence
#
#
def gaussian_kernel(image_shape, sigma):
    # kernel_size = int(2 * np.ceil(2 * sigma) + 1)

    kernel = gaussian_filter(np.zeros(image_shape), sigma=sigma)

    return kernel / np.sum(kernel)
#

def calc_supp(kernel,curr_conf):
    pass


def update_confidence(image, confidence, support):
    next_confidence = np.zeros_like(confidence)
    confidence_sum = np.zeros_like(confidence)
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            next_confidence[i][j] = confidence[i][j] * support[i][j]
            confidence_sum[i][j] += next_confidence[i][j]
    return next_confidence


def update_label(conff):
    pass

def relaxation_labeling(image,scale_factor,epsilon):
    scaled_image,base_pixels, initial_confidence = upscale(image, scale_factor, 0.5)
    # # ____
    kernel = gaussian_kernel(scaled_image.shape, scale_factor)
    curr_conf = initial_confidence
    k = 0  # counts the iteration number
    while True:
        support = calc_supp(kernel,curr_conf)
        next_conf = update_confidence(scaled_image,curr_conf, support)
        diff = np.linalg.norm(curr_conf.values() - next_conf.values()) #TODO: need to think about avg measurement?
        curr_conf = next_conf
        k += 1
        if diff < epsilon:
            break
    return update_label(next_conf)



def upsample_image(input_image, target_resolution):
    # Define the target resolutions
    HD_RESOLUTION = (1280, 720)
    FHD_RESOLUTION = (1920, 1080)
    QHD_RESOLUTION = (2560, 1440)
    h,w,c = input_image.shape
    DOUBLE = (4*w,4*h)

    # Get the input image dimensions
    height, width, channels = input_image.shape

    # Define the interpolation method
    interpolation_method = cv2.INTER_CUBIC

    # Upscale the image based on the target resolution
    if target_resolution == "HD":
        output_image = cv2.resize(input_image, HD_RESOLUTION, interpolation=interpolation_method)
    elif target_resolution == "FHD":
        output_image = cv2.resize(input_image, FHD_RESOLUTION, interpolation=interpolation_method)
    elif target_resolution == "QHD":
        output_image = cv2.resize(input_image, QHD_RESOLUTION, interpolation=interpolation_method)
    elif target_resolution == "D":
        output_image = cv2.resize(input_image, DOUBLE, interpolation=interpolation_method)
    else:
        print("Invalid target resolution. Please enter 'HD', 'FHD', or 'QHD'.")
        return None

    return output_image





if __name__ == '__main__':

    input_image_path = "nadal.jpg"
    target_resolution = "QHD"
    # Load the input image
    input_image = cv2.imread(input_image_path)
    input_image = cv2.resize(input_image, (200,200), interpolation=cv2.INTER_AREA)
    output1_image_path = f"downsampled_{target_resolution.lower()}_image.jpg"
    cv2.imwrite(output1_image_path, input_image)

    # Upsample the image
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    scale_factor = 10
    good = relaxation_labeling(gray_image, scale_factor,0.5)
    # upsampled_image = upsample_image(input_image, target_resolution)
    # # Save the upsampled image
    # if upsampled_image is not None:
    #     output_image_path = f"upsampled_{target_resolution.lower()}_image.jpg"
    #     cv2.imwrite(output_image_path, upsampled_image)
    #     print(f"Upsampled image saved as {output_image_path}")
    # else:
    #     print("Failed to upsample the image.")
    #
    # denoised_image = relaxation_labeling(upsampled_image, max_iterations=15, compatibility_factor=0.8)
    output_image_path = f"deno_upscale_{scale_factor}__image.jpg"
    cv2.imwrite(output_image_path, good)
    print("before:", gray_image.shape)
    print("after:", good.shape)
    # print("Denoised image saved as", output_image_path)


