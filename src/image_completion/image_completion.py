import numpy as np
import matplotlib.pyplot as plt


def fourier_predict_polygon(image):
    # Create a mask to ignore points inside the polygon
    mask = np.ones_like(image.gray_img, dtype=bool)
    if image.polygon_closed:
        polygon_points = np.array(image.clicked_points)
        if len(polygon_points) > 0:  # Check if polygon_points is not empty
            min_x, min_y = polygon_points.min(axis=0)
            max_x, max_y = polygon_points.max(axis=0)
            for i in range(int(min_y), int(max_y) + 1):
                for j in range(int(min_x), int(max_x) + 1):
                    if image.point_inside_polygon(j, i, polygon_points):
                        mask[i, j] = False

    # Apply Fourier transform
    f_transform = np.fft.fft2(image.gray_img * mask)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    # Apply inverse Fourier transform
    f_shift_back = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_shift_back)
    img_back = np.abs(img_back)

    # Plot the predicted values inside the polygon
    plt.imshow(img_back, cmap='gray')
    plt.axis('off')
    plt.title('Predicted Values Inside Polygon')
    plt.show()
