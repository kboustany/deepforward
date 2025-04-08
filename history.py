"""A class for storing and plotting model training history."""

import matplotlib.pyplot as plt


class History:
    """An object which records and displays the results of model training."""

    def __init__(self):
        self._data = [{}, {}, {}, {}]
        self._epochs = 0

    def summarize(self):
        """Summarize final losses and metrics, and plot training history."""
        print(f"Final training loss: {self._data[0][self._epochs]:.4f}      "
              f"Final validation loss: {self._data[1][self._epochs]:.4f}\n"
              f"Final training metric: {self._data[2][self._epochs]:.4f}    "
              f"Final validation metric: {self._data[3][self._epochs]:.4f}")
        self._plot()

    def update(self, data):
        """Update training data after training for one epoch."""
        self._epochs += 1
        for index, datum in enumerate(data):
            self._data[index][self._epochs] = datum

    def _plot(self):
        """Utility for configuring training history plots."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self._data[0].keys(),
                self._data[0].values(),
                label="Training Loss",
                color='red',
                linestyle="-")
        ax.plot(self._data[1].keys(),
                self._data[1].values(),
                label="Validation Loss",
                color='blue',
                linestyle="-")
        ax.plot(self._data[2].keys(),
                self._data[2].values(),
                label="Training Metric",
                color='red',
                linestyle="--")
        ax.plot(self._data[3].keys(),
                self._data[3].values(),
                label="Validation Metric",
                color='blue',
                linestyle="--")
        ax.axis([1, self._epochs, 0, 1.0])
        ax.set_xlabel("Epoch", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid()
        ax.legend()
        plt.show()
