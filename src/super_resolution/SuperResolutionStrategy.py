from abc import ABC, abstractmethod

import numpy as np


class SuperResolutionStrategy:
    """
    abstract Strategy class, which defines the interface for SuperResolutionStrategy
    """
    @abstractmethod
    def upscale(self, image, scale_factor):
        raise NotImplementedError()




class SuperResolutionContext:
    """
    Context class for the Super Resolution Strategy
    """
    def __init__(self, strategy):
        self.strategy = strategy

    def upscale(self, image, scale_factor):
        return self.strategy.upscale(image, scale_factor)
