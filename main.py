import numpy as np
import nnfs # Used to replicate the datasets used in the course
from nnfs.datasets import spiral_data

nnfs.init()
np.random.seed(0)


# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Rectified Linear Unit
class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_SoftMax:
    def forward(self, inputs):
        ex_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = ex_vals / np.sum(ex_vals, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: # Must be scalar values
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods


X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLu()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_SoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_func = Loss_CategoricalCrossentropy()
loss = loss_func.calculate(activation2.output, y)
print("Loss:", loss)


