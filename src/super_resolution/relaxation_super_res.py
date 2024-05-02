import numpy as np
import math
from scipy.signal import convolve2d

class RelaxationSuperRes:
    def __init__(self, kernel = None):
        if isinstance(kernel, np.ndarray):
            self.comp_kernel = kernel
        else:
            self.comp_kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]])

    def fit(self, image, base_pixels, n_buckets):
        self.image = image
        self.base_pixels = base_pixels
        self.n_buckets = n_buckets
        self._calc_initial_confidence()
        return self

    def _calc_initial_confidence(self):
        # print((self.image.shape[0],self.image.shape[1],2*self.n_buckets+1))
        h,w = self.image.shape
        self.initial_confidence = np.ones((h,w,2*self.n_buckets+1))
        for i in range(h):
            for j in range(w):
                for k,v in enumerate(self.initial_confidence[i,j]):

                    if (i,j) in self.base_pixels:
                        if k==self.n_buckets:
                            self.initial_confidence[i,j,k] = 0.8
                        else:
                            self.initial_confidence[i, j,k] = 0.1
                    else:

                        distance = abs(k - self.n_buckets) # interpolation_idx
                        if distance != 0:
                            sigma = self.n_buckets / math.sqrt(-2 * np.log(0.125))
                        else:
                            sigma = 2
                        self.initial_confidence[i,j,k] = 0.8 * np.exp(-0.5 * (distance/sigma)**2)
    
    def _calc_supp(self, confidence):
        return np.array([convolve2d(confidence[:, :, label], self.comp_kernel, mode='same') for label in
                     range(2*self.n_buckets+1)]).transpose(1, 2, 0)
    
    def _update_confidence(self, confidence, support):
        next_confidence = np.ones_like(confidence)
        confidence_sum = np.zeros_like(self.image)
        height, width = self.image.shape
        for i in range(height):
            for j in range(width):
                next_confidence[i][j] = confidence[i][j] * support[i][j]
                next_confidence[i][j] /= np.sum(next_confidence[i][j])
        return next_confidence , np.sum(confidence_sum)

    def _calc_label(self, confidence):
        final_img = self.image.copy().astype(np.uint8)
        optimal_indices = (np.argmax(confidence, axis=-1) - self.n_buckets).astype(np.uint8)
        final_img+=optimal_indices
        return final_img
    
    def predict(self, epsilon):
        curr_conf = self.initial_confidence
        k = 0  # counts the iteration number
        current_change =1
        while True:
            print("relaxation iter:", k)
            support = self._calc_supp(curr_conf)
            # support = calc_support(scaled_image, list(range(-10,11,1)),curr_conf, kernel)
            next_conf,next_change = self._update_confidence(curr_conf, support)
            diff = np.linalg.norm(curr_conf - next_conf) #TODO: need to think about avg measurement?
            k += 1
            print(diff)
            if diff< epsilon: #abs(current_change - next_change)
                break
            else:
                curr_conf = next_conf
                current_change = next_change
        # print(next_conf)
        return self._calc_label(next_conf)