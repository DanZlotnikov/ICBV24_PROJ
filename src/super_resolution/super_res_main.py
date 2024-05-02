import os
import cv2
import imageio
from src.super_resolution.relaxation_super_res import *
from src.super_resolution.upscaling_funcs import *

uploads_directory = '../uploads'
processed_directory = '../processed'

g5x5 = np.array([[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
                    [0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621],
                    [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
                    [0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621],
                    [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]
                    ], dtype=np.float32)

g3x3 = np.array([[0.0751136, 0.1238414, 0.0751136],
                    [0.1238414, 0.2041799, 0.1238414],
                    [0.0751136, 0.1238414, 0.0751136]
                    ], dtype=np.float32)

l3x3 = np.array([[0.0625, 0.125, 0.0625],
                    [0.125,  0.25,  0.125],
                    [0.0625, 0.125, 0.0625]])

l5x5 = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.08, 0.08, 0.08, 0.04],
                    [0.04, 0.08, 0.16, 0.08, 0.04],
                    [0.04, 0.08, 0.08, 0.08, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04]])

o3x3 = np.ones((3,3))/9

o5x5 = np.ones((5,5))/25

def super_res(method, image_name, scale, bins):
    full_path = os.path.join(uploads_directory, image_name)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    print("super_res ", method, image_name, scale, bins)
    kernel = None
    if method == 'g3x3':
        kernel = g3x3
    elif method == 'g5x5':
        kernel = g5x5
    elif method == 'l3x3':
        kernel = l3x3
    elif method == 'l5x5':
        kernel = l5x5
    elif method == 'o3x3':
        kernel = o3x3
    elif method == 'o5x5':
        kernel = o5x5

    model = RelaxationSuperRes(kernel)
    upscaled_image, base_pixels = relaxation_upscale(image, scale)
    model.fit(upscaled_image, base_pixels, bins)
    processed_image = model.predict(0.001)
    new_path = os.path.join(processed_directory, "processed_"+image_name)
    print("new_path : " , new_path)
    imageio.imwrite(new_path, processed_image)
    return True

