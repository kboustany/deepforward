"""A set of classes representing model layers for a feedforward neural net."""

from deepforward.activations import *

activations = {'identity': Identity, 'softmax': Softmax, 'relu': ReLU}

def add_intercept(input):
    """Global utilitiy for adding column of ones to input."""
    return np.hstack((np.ones((len(input), 1)), input))


class Layer:
    """A base class for a layer in a feedforward neural network."""

    def __init__(self, columns):
        self._columns = columns
        self._rows = None
        self._trainable = False

    def is_trainable(self):
        """Determine if layer has weights to be updated during training."""
        return self._trainable
    
    def predict(self, input):
        """Return output of layer for use in normal prediction."""
        raise NotImplementedError("Must be implemented by subclass.")   

    def forward(self, input):
        """Return output of layer with special training-only behaviors."""
        raise NotImplementedError("Must be implemented by subclass.")
    
    def initialize(self):
        """Initialize layer weights if applicable."""
        raise NotImplementedError("Must be implemented by subclass.")
    
    def clear(self):
        """Clear additional layer data accumulated during training."""
        raise NotImplementedError("Must be implemented by subclass.")

    def get_columns(self):
        """Return number of neurons in layer."""
        return self._columns
    
    def set_rows(self, rows):
        """Set accepted input size for layer."""
        self._rows = rows


class Input(Layer):
    """An input layer in a feedforward neural network."""

    def __init__(self, columns):
        super().__init__(columns+1)
        self._rows = columns

    def predict(self, input):
        return add_intercept(input)

    def forward(self, input):
        return self.predict(input)
    
    def initialize(self):
        pass

    def clear(self):
        pass


class Dense(Layer):
    """A dense layer in a feedforward neural network."""

    def __init__(self, columns, activation='identity'):
        super().__init__(columns)
        self._activation = activations[activation]
        self._trainable = True
        self._input = None
        self._weights = None

    def predict(self, input):
        return self._activation.forward(input @ self._weights)

    def forward(self, input):
        self._input = input
        return self.predict(input)

    def initialize(self):
        scale = self._activation.initializer(self._rows, self._columns)
        size = (self._rows, self._columns)
        rng = np.random.default_rng(42)
        self._weights = rng.normal(scale=scale, size=size)

    def backward(self, error_weights):
        """Return gradient and error weights during backpropagation."""
        derivative = self._activation.backward(self._input @ self._weights)
        error = error_weights * derivative
        return self._input.T @ error, error @ self._weights.T
    
    def update_weights(self, update):
        """Update layer weights during backpropagataion."""
        self._weights += update

    def clear(self):
        self._input = None


class Dropout(Dense):
    """A dense layer with dropout in a feedforward neural network."""

    def __init__(self, columns, activation='identity', drop=0.2):
        super().__init__(columns, activation)
        self._keep = 1 - drop
        self._rng = np.random.default_rng(42)
        self._mask = None
        self._masked = None

    def forward(self, input):
        self._input = input
        self._mask = self._rng.binomial(1, self._keep, self._weights.shape)
        self._masked = self._weights * self._mask / self._keep
        return self._activation.forward(input @ self._masked)

    def backward(self, error_weights):
        derivative = self._activation.backward(self._input @ self._masked)
        error = error_weights * derivative
        return self._input.T @ error, error @ self._masked.T
