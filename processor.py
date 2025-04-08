"""A class of utilities for preprocessing training data."""

from math import ceil
from numpy import mean, std

def mean_std(input, flag):
    """Global utility for extracting mean and standard deviation."""
    return (mean(input, axis=0), std(input, axis=0)) if flag else None


class Processor:
    """A processor for splitting and scaling training data."""

    def __init__(self):
        self._X_scaler = None
        self._Y_scaler = None

    def scale_X(self, input, inverse=False):
        """Normalize input values to mean zero and standard deviation one.
        
        Inverse normalization returned if inverse=True.
        """
        if self._X_scaler:
            if inverse:
                return (self._X_scaler[1] * input) + self._X_scaler[0]
            return (input - self._X_scaler[0]) / self._X_scaler[1]
        return input
    
    def scale_Y(self, input, inverse=False):
        """Normalize output values to mean zero and standard deviation one.
        
        Inverse normalization returned if inverse=True.
        """
        if self._Y_scaler:
            if inverse:
                return (self._Y_scaler[1] * input) + self._Y_scaler[0]
            return (input - self._Y_scaler[0]) / self._Y_scaler[1]
        return input
    
    def call(self, X, Y, validation, scale_X, scale_Y):
        """Primary method called by model during data preprocessing.
        
        Training and validation sets created and normalized, then returned.
        """
        X_train, Y_train, X_valid, Y_valid = self._split(X, Y, validation)
        self._set_scalers(X_train, Y_train, scale_X, scale_Y)
        train_set = self.scale_X(X_train), self.scale_Y(Y_train)
        valid_set  = self.scale_X(X_valid), self.scale_Y(Y_valid)
        return train_set, valid_set
    
    def _split(self, X, Y, validation):
        """Split full training data into training and validation sets.
        
        validation is float between 0 and 1.
        """
        split = ceil(len(X) * (1 - validation))
        X_train, X_valid = X[:split], X[split:]
        Y_train, Y_valid = Y[:split], Y[split:]
        return X_train, Y_train, X_valid, Y_valid

    def _set_scalers(self, X_train, Y_train, scale_X, scale_Y):
        """Store scaler attributes for future model predictions."""
        self._X_scaler = mean_std(X_train, scale_X)
        self._Y_scaler = mean_std(Y_train, scale_Y)
