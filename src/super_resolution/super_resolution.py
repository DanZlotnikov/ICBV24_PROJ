import math

import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from scipy.signal import convolve2d
from SuperResolutionStrategy import SuperResolutionContext
from RelaxationLabelingStrategy import RelaxationLabelingStrategy


def calc_initial_confidence(scaled_image_shape,base_pixels,n_buckets):
    initial_confidence = np.ones((scaled_image_shape[0],scaled_image_shape[1],2*n_buckets+1))
    h,w = scaled_image_shape
    interpolation_idx = n_buckets
    for i in range(h):
        for j in range(w):
            for k,v in enumerate(initial_confidence[i,j]):

                if (i,j) in base_pixels:
                    if k==n_buckets:
                        initial_confidence[i,j,k] = 0.99
                    else:
                        initial_confidence[i, j,k] = 0.001
                else:

                    distance = abs(k - interpolation_idx)
                    if distance != 0:
                        sigma = distance / math.sqrt(-2 * np.log(0.8))
                    else:
                        sigma = 2
                    initial_confidence[i,j,k] = 0.8 * np.exp(-0.5 * (distance/sigma)**2)

    return initial_confidence
# def upscale(image, scale_factor):
#     """
#     upscales the image by the scale factor using interpoltion to assign the new values to the upscaled image(initial labels
#     :param image:
#     :param scale_factor:
#     :return:
#     """
#     height, width = image.shape
#
#     # Create an upscaled image with the same dimensions
#     n_img = np.resize(image,( height*scale_factor,width*scale_factor))
#     upscaled_image = np.zeros_like(n_img, dtype=np.uint8)
#     base_pixels = {}
#     for y in range(height):
#         for x in range(width):
#             for dy in range(scale_factor):
#                 for dx in range(scale_factor):
#                     if dx == 0 and dy == 0:
#                         upscaled_image[y * scale_factor + dy, x * scale_factor + dx] = image[y, x]
#                         base_pixels[(y, x)] = image[y, x]
#                     else:
#                         weight_x = dx / scale_factor
#                         weight_y = dy / scale_factor
#
#                         upscaled_image[y * scale_factor + dy, x * scale_factor + dx] = (
#                                 (1 - weight_x) * (1 - weight_y) * image[y, x] +
#                                 weight_x * (1 - weight_y) * image[y, min(x + 1, width - 1)] +
#                                 (1 - weight_x) * weight_y * image[min(y + 1, height - 1), x] +
#                                 weight_x * weight_y * image[min(y + 1, height - 1), min(x + 1, width - 1)]
#                         )
#
#     return upscaled_image, base_pixels


def calc_supp(kernel, curr_conf,n_buckets):
    return np.array([convolve2d(curr_conf[:, :, label], kernel, mode='same') for label in
                     range(2*n_buckets+1)]).transpose(1, 2, 0)


def update_confidence(shape, confidence, support):
    next_confidence = np.ones_like(confidence)
    confidence_sum = np.zeros_like(confidence)
    height, width = shape
    for i in range(height):
        for j in range(width):
            next_confidence[i][j] = confidence[i][j] * support[i][j]
            confidence_sum[i][j] += next_confidence[i][j]
    return next_confidence/ confidence_sum , np.sum(confidence_sum)


def calc_label(scaled_img,conff,n_buckets):
    final_img = scaled_img.copy().astype(np.uint8)
    optimal_indices = (np.argmax(conff, axis=-1) - n_buckets).astype(np.uint8)
    final_img-=optimal_indices

    return final_img


def relaxation_labeling(image,scale_factor,n_buckets,epsilon):
    scaled_image,base_pixels = upscale(image, scale_factor)
    # # ____
    # kernel =  np.ones((5, 5))
    # linear kernel:
    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]])
    curr_conf = calc_initial_confidence(scaled_image.shape,base_pixels,n_buckets)
    print(curr_conf)
    k = 0  # counts the iteration number
    current_change =1
    while True:
        print("relaxation iter:", k)
        support = calc_supp(kernel,curr_conf,n_buckets)
        # support = calc_support(scaled_image, list(range(-10,11,1)),curr_conf, kernel)
        next_conf,next_change = update_confidence(scaled_image.shape,curr_conf, support)
        diff = np.linalg.norm(curr_conf - next_conf) #TODO: need to think about avg measurement?
        k += 1
        print(diff)
        if  diff< epsilon: #abs(current_change - next_change)
            break
        else:
          curr_conf = next_conf
          current_change = next_change
    # print(next_conf)
    return calc_label(scaled_image,next_conf,n_buckets)



def upsample_image(input_image, target_resolution):
    # Define the target resolutions
    HD_RESOLUTION = (1280, 720)
    FHD_RESOLUTION = (1920, 1080)
    QHD_RESOLUTION = (2560, 1440)
    h,w,c = input_image.shape
    DOUBLE = (4*w,4*h)

    height, width, channels = input_image.shape

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


def img_prep(input_image_path, downscale_dim):
    """
    gets an image, downsample it to the desired lower resolution turn it into greyscale and saves it for future use.
    uses cv2 resize to get downscaled image
    :param input_image:
    :param downscale_dim:
    :return:
    """
    input_image = cv2.imread(input_image_path)
    img = cv2.resize(input_image, downscale_dim, interpolation=cv2.INTER_AREA)
    gray_downscaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    saved_img_path = f"downscaled_greyscaled_{downscale_dim}_{input_image_path}"
    cv2.imwrite(saved_img_path, gray_downscaled_image)
    return gray_downscaled_image




def run_super_resolution(upscaling_strategy, input_image_path, scale_factor,n_buckets,epsilon):
    downscaled_dim = (200,200)
    init_img = img_prep(input_image_path,downscaled_dim)  # init img - downsampled, grey_scaled img
    scaled_image,base_pixels = upscaling_strategy.upscale(init_img, scale_factor)  # insert Strategy use here
    preprocessed_image = scaled_image.copy() # preprocessed_image = image after the upscale
    processed_image = relaxation_labeling(scaled_image,scale_factor,n_buckets,epsilon) # img after relaxation labeling
    cv2.imwrite(f"upscaled_image_sf_{scale_factor}_{input_image_path}", preprocessed_image)
    cv2.imwrite(f"relaxation_labeling_{scale_factor}_n_buckets_{n_buckets}_eps_{epsilon}_{input_image_path}", processed_image)







if __name__ == '__main__':

    input_image_path = "lenna.png"
    interpolation_strategy = RelaxationLabelingStrategy()
    super_resolution_strategy = SuperResolutionContext(interpolation_strategy)
    run_super_resolution(super_resolution_strategy,input_image_path,scale_factor=10,n_buckets=10,epsilon=0.1)

    # Load the input image
    # input_image = cv2.imread(input_image_path)
    # # image after downsample to 200X200
    # input_image = cv2.resize(input_image, (200,200), interpolation=cv2.INTER_AREA)
    # output1_image_path = f"downsampled_lenna_image.jpg"
    # cv2.imwrite(output1_image_path, input_image)

    # #gray downsampled:
    # downsampled_gray_path = f"downsampled_gray_lenna_image.jpg"
    # gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(downsampled_gray_path, gray_image)

    # image after upscaling by scale factor of 10:
    # scale_factor = 10
    # img_after_upscaling, _ = upscale(gray_image, scale_factor)
    # after_upscaling = f"upsampled_with_factor_{scale_factor}_noise_lenna_image.jpg"
    # cv2.imwrite(after_upscaling, img_after_upscaling)
    #
    # # denoised upsampled image:
    # eps = 0.01
    # n_buckets = 20 # +val -val from the interpolation values, should be even number
    # good = relaxation_labeling(gray_image, scale_factor,n_buckets,eps)
    # output_image_path = f"denoised_upscaled_{scale_factor}_eps={eps}_lenna_image.jpg"
    # cv2.imwrite(output_image_path, good)
    #
    # print("before:", gray_image.shape)
    # print("after:", good.shape)
    # print("Denoised image saved as", output_image_path)


