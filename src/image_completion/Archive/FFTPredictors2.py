from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.timeseries import TimeSeries
from typing import Optional, Callable, Sequence, Union, Tuple, List
import numpy as np
from darts.utils.missing_values import fill_missing_values
from typing import Callable, Optional

from darts.timeseries import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.models.forecasting.fft import _find_relevant_timestamp_attributes, _crop_to_match_seasons
import cv2

class FFT2(GlobalForecastingModel):
    def __init__(
        self,
        nr_freqs_to_keep: Optional[int] = 10,
        required_matches: Optional[set] = None,
        trend: Optional[str] = None,
        trend_poly_degree: int = 3,
    ):
        """Fast Fourier Transform Model

        This model performs forecasting on a TimeSeries instance using FFT, subsequent frequency filtering
        (controlled by the `nr_freqs_to_keep` argument) and  inverse FFT, combined with the option to detrend
        the data (controlled by the `trend` argument) and to crop the training sequence to full seasonal periods
        Note that if the training series contains any NaNs (missing values), these will be filled using
        :func:`darts.utils.missing_values.fill_missing_values()`.

        Parameters
        ----------
        nr_freqs_to_keep
            The total number of frequencies that will be used for forecasting.
        required_matches
            The attributes of pd.Timestamp that will be used to create a training sequence that is cropped at the
            beginning such that the first timestamp of the training sequence and the first prediction point have
            matching phases. If the series has a yearly seasonality, include `month`, if it has a monthly
            seasonality, include `day`, etc. If not set, or explicitly set to None, the model tries to find the
            pd.Timestamp attributes that are relevant for the seasonality automatically.
        trend
            If set, indicates what kind of detrending will be applied before performing DFT.
            Possible values: 'poly', 'exp' or None, for polynomial trend, exponential trend or no trend, respectively.
        trend_poly_degree
            The degree of the polynomial that will be used for detrending, if `trend='poly'`.

        Examples
        --------
        Automatically detect the seasonal periods, uses the 10 most significant frequencies for
        forecasting and expect no global trend to be present in the data:

        >>> FFT(nr_freqs_to_keep=10)

        Assume the provided TimeSeries instances will have a monthly seasonality and an exponential
        global trend, and do not perform any frequency filtering:

        >>> FFT(required_matches={'month'}, trend='exp')

        Simple usage example, using one of the dataset available in darts
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import FFT
        >>> series = AirPassengersDataset().load()
        >>> # increase the number of frequency and use a polynomial trend of degree 2
        >>> model = FFT(
        >>>     nr_freqs_to_keep=20,
        >>>     trend= "poly",
        >>>     trend_poly_degree=2
        >>> )
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[471.79323146],
               [494.6381425 ],
               [504.5659999 ],
               [515.82463265],
               [520.59404623],
               [547.26720705]])

        .. note::
            `FFT example notebook <https://unit8co.github.io/darts/examples/03-FFT-examples.html>`_ presents techniques
            that can be used to improve the forecasts quality compared to this simple usage example.
        """
        super().__init__()
        self.nr_freqs_to_keep = nr_freqs_to_keep
        self.required_matches = required_matches
        self.trend = trend
        self.trend_poly_degree = trend_poly_degree

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[List[int]],
        Optional[List[int]],
    ]:
        return None, None, False, False, None, None

    @property
    def extreme_lags(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
    ]:
        # TODO: LocalForecastingModels do not yet handle extreme lags properly. Especially
        #  TransferableFutureCovariatesLocalForecastingModel, where there is a difference between fit and predict mode)
        #  do not yet. In general, Local models train on the entire series (input=output), different to Global models
        #  that use an input to predict an output.
        return -self.min_train_series_length, -1, None, None, None, None
    
    @property
    def supports_multivariate(self) -> bool:
        return True

    def _exp_trend(self, x) -> Callable:
        """Helper function, used to make FFT model pickable."""
        return np.exp(self.trend_coefficients[1]) * np.exp(
            self.trend_coefficients[0] * x
        )

    def _poly_trend(self, trend_coefficients) -> Callable:
        """Helper function, for consistency with the other trends"""
        return np.poly1d(trend_coefficients)

    def _null_trend(self, x) -> Callable:
        """Helper function, used to make FFT model pickable."""
        return 0

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]]):
        series = fill_missing_values(series)
        super().fit(series)
        self._assert_univariate(series)
        series = self.training_series

        # determine trend
        if self.trend == "poly":
            self.trend_coefficients = np.polyfit(
                range(len(series)), series.univariate_values(), self.trend_poly_degree
            )
            self.trend_function = self._poly_trend(self.trend_coefficients)
        elif self.trend == "exp":
            self.trend_coefficients = np.polyfit(
                range(len(series)), np.log(series.univariate_values()), 1
            )
            self.trend_function = self._exp_trend
        else:
            self.trend_coefficients = None
            self.trend_function = self._null_trend

        # subtract trend
        detrended_values = series.univariate_values() - self.trend_function(
            range(len(series))
        )
        detrended_series = TimeSeries.from_times_and_values(
            series.time_index, detrended_values
        )

        # crop training set to match the seasonality of the first prediction point
        if self.required_matches is None:
            curr_required_matches = _find_relevant_timestamp_attributes(
                detrended_series
            )
        else:
            curr_required_matches = self.required_matches
        cropped_series = _crop_to_match_seasons(
            detrended_series, required_matches=curr_required_matches
        )

        # perform dft
        self.fft_values = np.fft.fft(cropped_series.univariate_values())

        # get indices of `nr_freqs_to_keep` (if a correct value was provided) frequencies with the highest amplitudes
        # by partitioning around the element with sorted index -nr_freqs_to_keep instead of sorting the whole array
        first_n = self.nr_freqs_to_keep
        if first_n is None or first_n < 1 or first_n > len(self.fft_values):
            first_n = len(self.fft_values)
        self.filtered_indices = np.argpartition(abs(self.fft_values), -first_n)[
            -first_n:
        ]

        # set all other values in the frequency domain to 0
        self.fft_values_filtered = np.zeros(len(self.fft_values), dtype=np.complex_)
        self.fft_values_filtered[self.filtered_indices] = self.fft_values[
            self.filtered_indices
        ]

        # precompute all possible predicted values using inverse dft
        self.predicted_values = np.fft.ifft(self.fft_values_filtered).real

        return self

    def predict(self, n: int, num_samples: int = 1, verbose: bool = False):
        super().predict(n, num_samples)
        trend_forecast = np.array(
            [self.trend_function(i + len(self.training_series)) for i in range(n)]
        )
        periodic_forecast = np.array(
            [self.predicted_values[i % len(self.predicted_values)] for i in range(n)]
        )
        return self._build_forecast_series(periodic_forecast + trend_forecast)
    

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

        self.left_model = FFT2(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(left)
        self.right_model = FFT2(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(np.flip(right, axis=1)) # Reverse the horizontal axis
        self.top_model = FFT2(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(np.transpose(top)) # Transpose the array
        self.bottom_model = FFT2(nr_freqs_to_keep, required_matches, trend, trend_poly_degree).fit(np.rot90(bottom, -1)) # Rotate the array 90 degrees clockwise

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
    
index = (100,50)
width, height = 20, 20
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
corrupted = img[index[0]:index[0]+height,index[1]:index[1]+width]
model = FFTRepair(img, weight_func='distance').fit(index, width, height, nr_freqs_to_keep=20, required_matches=None, trend=None, trend_poly_degree=3)
predicted= model.predict()