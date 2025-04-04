"""A class representing a feedforward neural network."""

from itertools import pairwise
from processor import Processor
from trainer import Trainer


class Model:
    """A feedforward neural network model for regression and classification."""

    # Constructor. -------------------------------------------------------------

    def __init__(self, layers):
        self._layers = layers
        for current, next in pairwise(self._layers):
            next.set_rows(current.get_columns()) # Set layer dimensions.
            next.initialize()                    # Initialize layer weights.
        self._processor = Processor()
        self._trainer = None
        self._preprocessed = False
        self._configured = False
        self._trained = False

    # Public accessors. --------------------------------------------------------
    
    def is_preprocessed(self):
        """Return True if training data has been preprocessed."""
        return self._preprocessed

    def is_configured(self):
        """Return True if model has been configured for learning task."""
        return self._configured

    def is_trained(self):
        """Return True if model was previously trained."""
        return self._trained
    
    def layers(self):
        """Return list of model layers."""
        return self._layers
    
    def predict(self, input):
        """Return model prediction from given input, with scaling if desired."""
        if not self.is_trained():
            raise AttributeError("Model is not trained.")
        prediction = self._processor.scale_X(input)
        for layer in self._layers:
            prediction = layer.predict(prediction)
        return self._processor.scale_Y(prediction, inverse=True)
    
    def evaluate(self, input, true):
        """Evaluate model performance on test set."""
        prediction = self.predict(input)
        loss, metric = self._trainer.evaluate(prediction, true)
        print(f"Test loss: {loss:.4f}    "
              f"Test metric: {metric:.4f}")

    # Public mutatators. -------------------------------------------------------

    def preprocess(self, X, Y, validation=0.1, scale_X=False, scale_Y=False):
        """Split full training data into training and validation sets.
        
        Scale training and validation sets for model training.
        """
        self._preprocessed = True
        return self._processor.call(X, Y, validation, scale_X, scale_Y)

    def configure(self, task, optimizer):
        """Configure model for regression or classification task.
        
        Fix optimizer to be used during model training.
        """
        self._trainer = Trainer(self, task, optimizer)
        self._configured = True

    def train(self, train_set, valid_set, epochs, batch_size=32):
        """Train model for specific number of epochs."""
        self._validate_training(epochs, batch_size, len(train_set[0]))
        self._trainer.call(train_set, valid_set, epochs, batch_size)
        self._trained = True

    # Nonpublic utilities. -----------------------------------------------------

    def _validate_training(self, epochs, batch_size, max_size):
        """Determine if model is properly initialized for training."""
        if not epochs >= 1:
            raise ValueError("Number of epochs must be positive.")
        if not 1 <= batch_size <= max_size:
            raise ValueError("Invalid batch size.")
        if not self.is_preprocessed():
            raise AttributeError("Model is not preprocessed.")
        if not self.is_configured():
            raise AttributeError("Model is not configured.")
