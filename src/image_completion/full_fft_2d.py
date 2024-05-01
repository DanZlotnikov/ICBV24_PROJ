import numpy as np

class FullFFT2D:
    def __init__(self, nr_freqs_to_keep: int = 10):
        """
        Initialize the FullFFT2D model.
        
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
        """
        Predict the missing part of the image.
        
        Returns:
        - (np.ndarray): Predicted values.
        """
        left = self.image[:, :x]
        right = self.image[:, x + w:]
        top = self.image[:y, :]
        bottom = self.image[y + h:, :]
        regions = [left, right, top, bottom]

        # Applying FFT on each region
        fft_results = [np.fft.fft2(region) for region in regions]
        
        # Determine the maximum size in each dimension
        max_rows = max(arr.shape[0] for arr in fft_results)
        max_cols = max(arr.shape[1] for arr in fft_results)
        fft_results_padded = [np.pad(arr, ((0, max_rows - arr.shape[0]), (0, max_cols - arr.shape[1])), mode='constant', constant_values=0) for arr in fft_results]

        # Compute magnitudes
        fft_magnitudes = [np.abs(fft) for fft in fft_results_padded]

        # Sum magnitudes to find common frequencies
        frequency_sum = np.sum(fft_magnitudes, axis=0)

        # Example reconstruction code using the average common frequencies (adjust as needed)
        average_fft = np.mean([fft * (frequency_sum > np.percentile(frequency_sum, 95)) for fft in fft_results_padded], axis=0)
        reconstructed_full  = np.fft.ifft2(average_fft).real

        reconstructed_part = reconstructed_full[x:x + w, y:y + h]

        return reconstructed_part.astype(int)