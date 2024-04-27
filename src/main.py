from src.image_completion import image_completion
import cv2 as cv

if __name__ == '__main__':
    img = cv.imread('./nadal.jpg', cv.IMREAD_GRAYSCALE)
    image_completion.complete_image(img)