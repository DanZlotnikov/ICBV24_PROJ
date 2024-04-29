# image_processing.py


def divide_into_subimages(image):
    M = image.shape[0] // 2
    N = image.shape[1] // 2
    subimages = [image[x:x+M, y:y+N] for x in range(0, image.shape[0], M) for y in range(0, image.shape[1], N)]


    return subimages
