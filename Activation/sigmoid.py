import numpy as np
from .activation import *

class Sigmoid(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        a = 1 / (1 + np.exp(-z))
        return a