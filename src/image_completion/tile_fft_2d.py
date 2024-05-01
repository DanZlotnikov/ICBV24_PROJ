import numpy as np
from src.image_completion.full_fft_2d import FullFFT2D

class TileFFT2D:
    def __init__(self, nr_freqs_to_keep: int = 10):
        """
        Initialize the TileFFT2D model.
        
        Parameters:
        - nr_freqs_to_keep (int): Number of highest magnitude frequencies to keep.
        """
        self.nr_freqs_to_keep = nr_freqs_to_keep

    def fit(self, image: np.ndarray):
        """
        Apply 2D FFT to the image and keep only the largest frequencies as specified by nr_freqs_to_keep.
        
        Parameters:
        - image (np.array): 2D array representing the grayscale image.
        """

        self.image = image
        return self
    
    def predict(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        updated_image = np.copy(self.image)
        num_tiles = 5
        diff_x = int(w / num_tiles)
        diff_y = int(h / num_tiles)

        for i in range(num_tiles):
            for j in range(num_tiles):
                start_x = x + i * diff_x
                end_x = x + (i + 1) * diff_x
                start_y = y + j * diff_y
                end_y = y + (j + 1) * diff_y
                reconstructed_part = FullFFT2D().fit(updated_image).predict(start_x, start_y, diff_x, diff_y)
                updated_image[start_x: end_x, start_y: end_y] = reconstructed_part

        return updated_image[x: x + w, y: y + h].astype(int)