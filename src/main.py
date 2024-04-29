from src.image import Image
from src.image_completion.image_completion import *

if __name__ == '__main__':
    image = Image('./nadal.jpg')
    image.plot()
    fourier_predict_polygon(image)
