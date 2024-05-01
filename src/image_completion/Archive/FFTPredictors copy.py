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

    def fit(self, image: np.ndarray):
        """
        Apply 2D FFT to the image and keep only the largest frequencies as specified by nr_freqs_to_keep.
        
        Parameters:
        - image (np.array): 2D array representing the grayscale image.
        """
        self.image = image
        self._models = [FFT(required_matches=self.required_matches, trend=self.trend, trend_poly_degree=self.trend_poly_degree) for _ in range(image.shape[0])]

        for i in range(image.shape[0]):
            self._models[i].fit(TimeSeries.from_values(image[i, :]))  # Reverse the horizontal axis if needed

        return self

    def predict(self, n: int, num_samples: int = 1, verbose: bool = False):
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
        return np.hstack([self._models[i].predict(n, num_samples, verbose).data_array() for i in range(self.image.shape[0])]).astype(int).reshape(self.image.shape)
        
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
        left = self.image[index[0]:index[0]+height, max(0,index[1]-width):index[1]]
        right = self.image[index[0]:index[0]+height, index[1]+width:min(index[1]+2*width, self.image.shape[1])]
        top = self.image[max(0, index[0]-height):index[0], index[1]:index[1]+width]
        bottom = self.image[index[0]+height:min(index[0]+2*height,self.image.shape[0]), index[1]:index[1]+width]

        self.left_model = FFT2D(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(left)
        self.right_model = FFT2D(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(np.flip(right, axis=1)) # Reverse the horizontal axis
        self.top_model = FFT2D(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(np.transpose(top)) # Transpose the array
        self.bottom_model = FFT2D(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(np.rot90(bottom, -1)) # Rotate the array 90 degrees clockwise

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

        left = self.left_model.predict(self.part_width, num_samples, verbose)
        right = np.flip(self.right_model.predict(self.part_height, num_samples, verbose),axis=1) # Reverse the horizontal axis
        top = np.transpose(self.top_model.predict(self.part_height, num_samples, verbose)) # Transpose the array
        bottom = np.rot90(self.bottom_model.predict(self.part_height, num_samples, verbose),1) # Rotate the array 90 degrees counterclockwise

        # Compute distance-based weights
        vertical_weights = np.linspace(1, 1e-3, self.part_height).reshape(-1, 1)
        horizontal_weights = np.linspace(1, 1e-3, self.part_width).reshape(1, -1)

        # Apply weights to predictions
        weighted_left = left * horizontal_weights
        weighted_right = right * np.flip(horizontal_weights, axis=1)
        weighted_top = top * vertical_weights
        weighted_bottom = bottom * np.flip(vertical_weights, axis=0)

        if self.weight_func == 'equal':
            combined_prediction = (left + right + top + bottom) / 4
        elif self.weight_func == 'distance':
            weighted_sum = (weighted_left + weighted_right + weighted_top + weighted_bottom)
            horizontal_weights_full = np.tile(horizontal_weights, (self.part_height, 1))
            vertical_weights_full = np.tile(vertical_weights, (1, self.part_width))

            weights_sum = horizontal_weights_full + np.flip(horizontal_weights_full, axis=1) + vertical_weights_full + np.flip(vertical_weights_full, axis=0)

            # weights_sum = (horizontal_weights + horizontal_weights + vertical_weights + vertical_weights)
            combined_prediction = weighted_sum / weights_sum
        else:
            raise ValueError('Invalid weight function')

        return combined_prediction.astype(int)