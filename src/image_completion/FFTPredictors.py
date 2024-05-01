import numpy as np
from darts.models import FFT
from darts.timeseries import TimeSeries
from typing import Optional

class FFT2D:
    def __init__(self, nr_freqs_to_keep: Optional[int] = 10,
        required_matches: Optional[set] = None,
        trend: Optional[str] = None,
        trend_poly_degree: int = 3,):
        """
        Initialize the FFT model for images.
        
        Parameters:
        - nr_freqs_to_keep (int): Number of highest magnitude frequencies to keep.
        """
        self.nr_freqs_to_keep = nr_freqs_to_keep
        self.required_matches = required_matches
        self.trend = trend
        self.trend_poly_degree = trend_poly_degree

    def fit(self, image: np.ndarray, reversed: bool = False):
        """
        Apply 2D FFT to the image and keep only the largest frequencies as specified by nr_freqs_to_keep.
        
        Parameters:
        - image (np.array): 2D array representing the grayscale image.
        """
        self.image = image
        self._horizontal_models = [FFT(required_matches=self.required_matches, trend=self.trend, trend_poly_degree=self.trend_poly_degree) for _ in range(image.shape[0])]
        self._vertical_models = [FFT(required_matches=self.required_matches, trend=self.trend, trend_poly_degree=self.trend_poly_degree) for _ in range(image.shape[1])]

        for i in range(image.shape[0]):
            self._horizontal_models[i].fit(TimeSeries.from_values(np.flip(image[i, :]) if reversed else image[i, :]))  # Reverse the horizontal axis if needed

        for j in range(image.shape[1]):
            self._vertical_models[j].fit(TimeSeries.from_values(np.flip(image[:, j]) if reversed else image[:, j]))  # Reverse the vertical axis if needed

        return self

    def predict(self, n: int, num_samples: int = 1, verbose: bool = False, horitonzal: bool = True):
        """
        Predict the next n values in the time series.
        
        Parameters:
        - n (int): Number of time steps to predict.
        - num_samples (int): Number of samples to generate.
        - verbose (bool): Whether to print the progress of the prediction.
        - horizontal (bool): Whether to predict horizontally or vertically.
        
        Returns:
        - (np.array): Predicted values.
        """
        if horitonzal:
            return np.array([self._horizontal_models[i].predict(n, num_samples, verbose) for i in range(self.image.shape[0])])
        else:
            return np.array([self._vertical_models[j].predict(n, num_samples, verbose) for j in range(self.image.shape[1])])
        
class FFTRepair:
    def __init__(self, image:np.ndarray, weight_func:str='equal'):
        """
        Initialize the FFTRepair model.
        
        Parameters:
        - image (np.array): 2D array representing the grayscale image.
        - weight_func (str): The weight function to use for combining the predictions.
        """
        self.image = image
        self.weight_func = weight_func

    def fit(self, index, width, height, nr_freqs_to_keep: Optional[int] = 10, required_matches: Optional[set] = None, trend: Optional[str] = None, trend_poly_degree: int = 3):
        """
        Fit the model to the corrupted patch.

        Parameters:
        - index (tuple): The top-left corner of the corrupted patch.
        - width (int): The width of the corrupted patch.
        - height (int): The height of the corrupted patch.
        - nr_freqs_to_keep (int): Number of highest magnitude frequencies to keep.
        - required_matches (set): Set of frequencies to keep.
        - trend (str): The trend to remove from the time series.
        - trend_poly_degree (int): The degree of the polynomial to fit the trend.
        """
        left = self.image[index[0]:index[0]+height, 0:index[1]]
        right = self.image[index[0]:index[0]+height, index[1]+width:]
        top = self.image[0:index[0], index[1]:index[1]+width]
        bottom = self.image[index[0]+height:, index[1]:index[1]+width]
        
        self.left_model = FFT2D(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(left, reversed=False)
        self.right_model = FFT2D(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(right, reversed=True)
        self.top_model = FFT2D(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(top, reversed=False)
        self.bottom_model = FFT2D(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(bottom, reversed=True)

        self.part_height = height
        self.part_width = width

        return self
    
    def predict(self, num_samples: int = 1, verbose: bool = False):
        """
        Predict the missing patch.
        
        Parameters:
        - num_samples (int): Number of samples to generate.
        - verbose (bool): Whether to print the progress of the prediction.
        """

        left = self.left_model.predict(self.part_width, num_samples, verbose, horitonzal=False)
        right = np.flip(self.right_model.predict(self.part_height, num_samples, verbose, horitonzal=False),axis=1) # Reverse the horizontal axis
        top = self.top_model.predict(self.part_height, num_samples, verbose, horitonzal=True)
        bottom = np.flip(self.bottom_model.predict(self.part_height, num_samples, verbose, horitonzal=True),axis=0) # Reverse the vertical axis

        if self.weight_func == 'equal':
            self._combine_predictions = lambda left, right, top, bottom: (left + right + top + bottom) / 4
        elif self.weight_func == 'distance':
            self._combine_predictions = lambda left, right, top, bottom: (left + right + top + bottom) / (np.abs(np.arange(self.part_height) - self.part_height // 2) + np.abs(np.arange(self.part_width) - self.part_width // 2) + 1)
        else:
            raise ValueError('Invalid weight function')

        return self._combine_predictions(left, right, top, bottom)