"""A set of classes modeling optimizers to be used during gradient descent."""

from numpy import sqrt


class Optimizer:
    """A base class for an optimizer used in gradient descent."""
    
    def __init__(self, rate): 
        self._rate = rate
        self._iteration = 0

    def set_layers(self, model):
        """Store model layers for optimizers which keep layer-wise data."""
        raise NotImplementedError("Must be implemented by subclass.")

    def _algorithm(self, layer, gradient, batch_size):
        """Algorithm used by optimizer to calculate weight update."""
        raise NotImplementedError("Must be implemented by subclass.")

    def update(self, layer, gradient, batch_size):
        """Increase iteration and returns weight update to trainer."""
        self._iteration += 1
        return self._algorithm(layer, gradient, batch_size)


class SGD(Optimizer):
    """A standard gradient descent optimizer."""

    def set_layers(self, model):
        pass

    def _algorithm(self, layer, gradient, batch_size):
        return -(self._rate / batch_size) * gradient


class Adam(Optimizer):
    """An adaptive moment estimation optimizer."""

    def __init__(self, rate): 
        super().__init__(rate)
        self._b1 = 0.9
        self._b2 = 0.999
        self._eps = 1e-7
        self._m = {}
        self._s = {}

    def set_layers(self, model):
        for layer in model.layers():
            if layer.is_trainable():
                self._m[layer] = 0
                self._s[layer] = 0

    def _algorithm(self, layer, gradient, batch_size):
        self._m[layer] = (self._b1 * self._m[layer]) \
            + (1 - self._b1) * gradient
        self._s[layer] = (self._b2 * self._s[layer]) \
            + (1 - self._b2) * (gradient * gradient)
        m = self._m[layer] / (1 - (self._b1 ** self._iteration))
        s = self._s[layer] / (1 - (self._b2 ** self._iteration))
        return -(self._rate / batch_size) * (m / sqrt(s + self._eps))
