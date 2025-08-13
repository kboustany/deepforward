# DeepForward
A deep learning library for training dense feedforward neural nets, built entirely on NumPy.

This project was originally meant for self-educational purposes. I thought writing a toy library would be a good way to
really learn the nuts and bolts of training deep networks. The code itself is my own work, but the interface is inspired
by that of Keras. Training models is done using the recursive backpropagation equations, hardcoded into the Trainer class,
as opposed to reverse autodiff.

The implementation of the library is rather efficient, as testing by training two deep models on the Fashion-MNIST dataset
using this library and the Keras library yielded very close training times, with my library even sometimes performing
significantly faster on my local machine.

# Setup

Since this was not really meant to be something people would use in lieu of Keras or PyTorch, I did not bother turning it into a package. This being said, you can still use inside a Colab notebook if you so choose. The process for doing so is simple and demonstrated in the example notebooks contained herein.
