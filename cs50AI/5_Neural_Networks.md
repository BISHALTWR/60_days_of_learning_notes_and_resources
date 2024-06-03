# Neural Networks

> Artificial Neural Network is a mathematical model for learning inspired by biological neural networks.

- Each neuron is capable of both receiving and sending electrical signals. Once the electrical input that a neuron receives crosses some threshold, the neuron activates, thus sending its electrical signal forward.

- Parallel of each neuron is a unit that is connected to other units:

## Activation function

> Activation functions help make decisions based on inputs, bias, and weights.

- Hypothesis function: `h(x₁, x₂) = w₀ + w₁x₁ + w₂x₂`, where `w₁` and `w₂` are weights that modify the inputs, and `w₀` is a constant, also called bias, modifying the value of the whole expression.

```markdown
# Step function

> Step function gives 0 before a certain threshold is reached and 1 after the threshold is reached.
```
![Step Function](step_function.png)

```markdown
# Logistic function

> gives as output any real number from 0 to 1, thus expressing graded confidence in its judgment.
```
![Logistic Sigmoid Function](logistic_sigmoid_function.png)

```markdown
# Rectified Linear Unit(ReLU)

- allows the output to be any positive value. If the value is negative, ReLU sets it to 0.
```
![ReLU Function](relu_function.png)

## Neural Network Structure

> A neural network can be thought of as a representation of the idea, where a function sums up inputs to produce an output.
A neural network is capable of infering knowledge about the structure of the network itself from the data.
## Gradient Descent

> Gradient descent is an algorithm for minimizing loss when training neural networks. 

Neural networks allow us to compute these weights based on the training data. To do this, we use the gradient descent algorithm, which works in the following way:

1. Start with a random choice of weights. This is our naive starting place, where we don’t know how much we should weight each input.
2. Repeat the following steps:
    - Calculate the gradient based on all data points that will lead to decreasing loss. Ultimately, the gradient is a vector (a sequence of numbers).
    - Update weights according to the gradient.

- The above algorithm requires to calculate the gradient based on all data points, which is computationally costly. There are two common ways to minimize cost:
    - Stochastic Gradient Descent (SGD) calculates the gradient based on one randomly chosen data point
    - Mini-Batch Gradient Descent uses a small random sample of data points, compromising between computation and accuracy.

- None of these solutions is perfect, and different solutions might be employed in different situations.

## Perceptron:

> Perceptron-based neural networks can only learn linear decision boundaries, effectively separating data with a straight line. However, when data is not linearly separable, multilayer neural networks are used to model data non-linearly.

![Perceptron](perceptron.png)

- This only works for linearly separable variables

## Multi layer neural networks

> A multilayer neural network is an artificial neural network with an input layer, an output layer, and at least one hidden layer.

- Through hidden layers, it is possible to model non-linear data.

## Backpropagation

> Backpropagation is the main algorithm used for training neural networks with hidden layers. It does so by starting with the errors in the output units, calculating the gradient descent for the weights of the previous layer, and repeating the process until the input layer is reached. 

To train a neural network using backpropagation, follow these steps:

1. **Calculate error for the output layer**: This is typically the difference between the network's output and the desired output.

2. **Propagate the error backwards**: Starting from the output layer and moving towards the input layer, propagate the error back one layer at a time. In other words, each layer sends its error to the preceding layer.

3. **Update the weights**: After the errors have been propagated all the way back to the input layer, use these errors to update the weights in the network. This is typically done using a method like gradient descent, where the weights are adjusted in the direction that most decreases the error.

Repeat these steps for many iterations, until the network's output is satisfactory.

## Overfitting

> Overfitting is the danger of modeling the training data too closely, thus failing to generalize to new data.

- One way to combat overfitting is by dropout.
    - In this technique, we temporarily remove random units that we select at learning phase. This prevents over-reliance on any one unit.

## TensorFlow