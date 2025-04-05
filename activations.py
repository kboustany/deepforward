"""A set of classes for activation functions to be used in a dense layer."""

import numpy as np


class Activation:
    """A base class for an activation function used in a layer."""

    def initializer(rows, columns):
        """Return scale for appropriate initialization of layer weights."""
        raise NotImplementedError("Must be implemented by subclass.")

    def forward(input):
        """Return output of activation for forward passes."""
        raise NotImplementedError("Must be implemented by subclass.")
    
    def backward(input):
        """Return derivative of activation for backward passes."""
        raise NotImplementedError("Must be implemented by subclass.")


class Identity(Activation):
    """A default identity activation for linear layers."""

    def initializer(rows, columns):
        return np.sqrt(2 / (rows + columns)) # Glorot initialization.

    def forward(input):
        return input
    
    def backward(input):
        return np.ones(input.shape)
        

class Softmax(Activation):
    """A softmax activation for outer layers of classification models."""

    def initializer(rows, columns):
        return np.sqrt(2 / (rows + columns)) # Glorot initialization.

    def forward(input):
        exp = np.exp(input - np.max(input))
        sum = exp.sum(axis=1, keepdims=True)
        return (1 / sum) * exp
    
    def backward(input):
        return np.ones(input.shape) # Convention for proper backpropagation.
            
    
class ReLU(Activation):
    """A rectified linear unit activataion for hidden layers of deep nets."""

    def initializer(rows, columns):
        return np.sqrt(2 / rows) # He initialization.

    def forward(input):
        return np.maximum(0, input)
    
    def backward(input):
        return np.sign(np.maximum(0, input))