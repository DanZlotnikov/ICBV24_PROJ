from src.image import Image
from src.image_completion.image_completion import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image = Image('./nadal.jpg')
    image.plot()

    # Call the function to divide the image into sub-images
    subimages = divide_into_subimages(image.img)

    # Process the sub-images further if needed
    for i, subimage in enumerate(subimages):
        # Perform operations on each sub-image
        # For example, you can display each sub-image
        plt.subplot(1, len(subimages), i + 1)
        plt.imshow(subimage, cmap='gray')
        plt.axis('off')
    plt.show()
