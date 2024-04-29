from src.image import Image
from src.image_completion.image_completion import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image = Image('./nadal.jpg')
    image.plot()

    # Call the function to divide the image into sub-images
    subimages = divide_into_subimages(image)

    # Process the sub-images further if needed
    num_subimages = len(subimages)
    num_rows = 3
    num_cols = num_subimages // num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    for i, subimage in enumerate(subimages):
        row = i // num_cols
        col = i % num_cols
        axes[col, row].imshow(subimage, cmap='gray')
        axes[col, row].axis('off')
    plt.show()