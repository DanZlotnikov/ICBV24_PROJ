from src.image import Image
import matplotlib.pyplot as plt
from src.image_completion import *
import cv2
from src.image_completion.full_fft_2d import FullFFT2D
from src.image_completion.axis_fft_2d import AxisFFT2D
from src.image_completion.image_completion import fft_complete

if __name__ == '__main__':
    fft_complete('Single', 'nadal.jpg', 100,100,100,100)
    
