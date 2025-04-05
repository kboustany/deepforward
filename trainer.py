"""A class containing a trainer used to train deep neural networks."""

from history import History
from tasks import *

tasks = {'regression': Regression, 'classification': Classification}

rng = np.random.default_rng(42)

def shuffle(data):
    """Global utility for shuffling data."""
    permutation = rng.permutation(len(data[0]))
    return data[0][permutation], data[1][permutation]

def batches(input, output, size):
    """Global utility for generating batches of data of given size."""
    i = 0
    while i + size < len(input):
        yield input[i:i+size], output[i:i+size]
        i += size
    yield input[i:], output[i:]


class Trainer:
    """A trainer for a feedforward neural network."""

    # Constructor. -------------------------------------------------------------

    def __init__(self, model, task, optimizer):
        self._model = model
        self._task = tasks[task]
        self._optimizer = optimizer
        self._optimizer.set_layers(model)
        self._history = History()

    # Public accessors. --------------------------------------------------------

    def evaluate(self, prediction, true):
        """Method called by model during evaluation."""
        loss = self._task.loss(prediction, true)
        metric = self._task.metric(prediction, true)
        return loss, metric
    
    # Public mutators. ---------------------------------------------------------

    def call(self, train_set, valid_set, epochs, batch_size):
        """Primary method called by model during training."""
        for _ in range(epochs):
            self._train_for_epoch(train_set, valid_set, batch_size)
        self._clear_layers()
        self._history.summarize()

    # Nonpublic utilities. -----------------------------------------------------

    def _train_for_epoch(self, train_set, valid_set, batch_size):
        """Train model for one epoch."""
        X_train, Y_train = shuffle(train_set)
        self._backpropagate(X_train, Y_train, batch_size)
        self._update_history(train_set, valid_set)

    def _backpropagate(self, X_train, Y_train, batch_size):
        """Update weights of model with backpropagation during epoch."""
        for X_batch, Y_batch in batches(X_train, Y_train, batch_size):
            prediction = self._forward_pass(X_batch)
            error_weights = self._task.error_weights(prediction, Y_batch)
            self._backward_pass(error_weights, batch_size)

    def _forward_pass(self, input):
        """Return result of one forward pass through model."""
        prediction = input
        for layer in self._model.layers():
            prediction = layer.forward(prediction)
        return prediction

    def _backward_pass(self, error_weights, batch_size):
        """Propagate gradient errors in one backward pass through model. """
        for layer in reversed(self._model.layers()):
            if layer.is_trainable():
                gradient, error_weights = layer.backward(error_weights)
                update = self._optimizer.update(layer, gradient, batch_size)
                layer.update_weights(update)

    def _update_history(self, train_set, valid_set):
        """Record training and validation losses and metrics for one epoch."""
        t_prediction = self._evaluation_pass(train_set[0])
        t_loss, t_metric = self.evaluate(t_prediction, train_set[1])
        v_prediction = self._evaluation_pass(valid_set[0])
        v_loss, v_metric = self.evaluate(v_prediction, valid_set[1])
        self._history.update([t_loss, v_loss, t_metric, v_metric])

    def _evaluation_pass(self, input):
        """Return result of one forward pass through model.
        
        Used for computing losses and metrics with no training-only behaviors.
        """
        prediction = input
        for layer in self._model.layers():
            prediction = layer.predict(prediction)
        return prediction
    
    def _clear_layers(self):
        """Clear model layers of additional stored data after training."""
        for layer in self._model.layers():
            layer.clear()