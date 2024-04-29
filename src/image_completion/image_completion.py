# image_processing.py


def divide_into_subimages(image):
    gray = image.gray_img
    x_markers = [0, int(image.rect_points[0][0]), int(image.rect_points[1][0]), gray.shape[1]]
    y_markers = [0, int(image.rect_points[0][1]), int(image.rect_points[1][1]), gray.shape[0]]

    sub_images = []

    # Loop through rows and columns to crop the image into 9 rectangles
    for i in range(0, len(y_markers) - 1):
        for j in range(0, len(x_markers) - 1):
            cropped = image.gray_img[y_markers[j]:y_markers[j + 1], x_markers[i]:x_markers[i + 1]]
            sub_images.append(cropped)

    return sub_images

