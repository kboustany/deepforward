"""A set of classes containing utilities for different learning tasks."""

import numpy as np


class Task:
    """A base class for a learning task to train a model on."""

    def loss(prediction, true):
        """Return appropriate loss value for current task."""
        raise NotImplementedError("Must be implemented by subclass.")
    
    def error_weights(prediction, true):
        """Return gradient errors to initialize recursive backpropagataion."""
        raise NotImplementedError("Must be implemented by subclass.")
    
    def metric(prediction, true):
        """Return performance metric for current task used for evaluation."""
        raise NotImplementedError("Must be implemented by subclass.")


class Regression(Task):
    """A Task class for regression models."""

    def loss(prediction, true):
        return np.square(prediction - true).sum() / len(true) # MSE.
    
    def error_weights(prediction, true):
        return 2 * (prediction - true) # Derivative of MSE.
    
    def metric(prediction, true):
        return np.sqrt(np.square(prediction - true).sum() / len(true)) # RMSE.

    
class Classification(Task):
    """A Task class for classification models."""

    def loss(prediction, true):
        prediction = np.clip(prediction, 1e-9, 1 - 1e-9)
        return -np.sum(np.log(prediction) * true) / len(true) # Cross-entropy.
    
    def error_weights(prediction, true):
        return prediction - true # Derivative of cross-entropy + softmax.
    
    def metric(prediction, true):
        hits = np.equal(np.argmax(prediction, axis=1), np.argmax(true, axis=1))
        return hits.sum() / len(true) # Prediction accuracy.