from src.image import Image
import matplotlib.pyplot as plt
from src.image_completion import *


if __name__ == '__main__':
    image = Image('./nadal.jpg')
    modified = fft_complete_tiles(image.gray_img, 800, 1200, 100, 100)
    plt.imshow(modified, cmap='gray')
    plt.axis('off')  # Turn off axis
    plt.show()


