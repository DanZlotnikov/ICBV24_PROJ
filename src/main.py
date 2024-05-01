from src.image import Image
import matplotlib.pyplot as plt
from src.image_completion import *

if __name__ == '__main__':
    modified = remove_rectangle('./nadal.jpg', 100, 100, 500, 500)
    plt.imshow(modified, cmap='gray')
    plt.axis('off')  # Turn off axis
    plt.show()


