# deepforward
A deep learning library for training dense feedforward neural nets, built entirely on NumPy.

This project was originally meant for self-educational purposes. I thought writing a toy library would be a good way to
really learn the nuts and bolts of training deep networks. The code itself is my own work, but the interface is inspired
by that of Keras. Training models is done using the recursive backpropagation equations, hardcoded into the Trainer class,
as opposed to reverse autodiff.

The implementation of the library is rather efficient, as testing by training two deep models on the Fashion-MNIST dataset
using this library and the Keras library yielded very close training times, with my library even sometimes performing
significantly faster on my local machine.

An obvious disclaimer: this library is not meant to perform on any level comparable to any of the standard professional
libraries available out there.
