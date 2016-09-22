__author__ = 'tan_nguyen', 'andrew_dumit'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from math import exp

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))


    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = actFun(self.z1, self.actFun_type)
        self.z2 = self.a1.dot(self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X)
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        # print(self.probs.shape)
        # print(y.shape)
        data_loss = -np.sum(np.log(self.probs[range(num_examples), y]))

        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2. * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        dW2 = 1./num_examples * self.a1.T.dot(delta3)
        db2 = 1./num_examples * delta3.sum(axis=0)
        dW1 = 1./num_examples * X.T.dot(np.multiply(diff_actFun(self.z1, self.actFun_type), delta3.dot(self.W2.T)))
        db1 = 1./num_examples * np.multiply(diff_actFun(self.z1, self.actFun_type), delta3.dot(self.W2.T)).sum(axis=0)
        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            dW1, dW2, db1, db2 = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


class DeepNeuralNetwork(NeuralNetwork):
    """
    This class builds and trains a deep neural network
    """
    def __init__(self, nn_input_dim, nn_output_dim, layers, layer_functions, reg_lambda=0.01, seed=0):
        """
        :param input_dim: dimension of the input
        :param output_dim: dimension of the output
        :param layers: list of integers that signify the size of each layer including input and output
        :param actFun_type: type of activation function
        :param reg_lambda: regularization term
        :param seed: What random seed to set
        """
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.layer_sizes = [self.nn_input_dim] + layers + [self.nn_output_dim]
        self.layers = [Layer(h, f) for h,f in zip(layers, layer_functions)]
        self.reg_lambda = reg_lambda
        # Create a W for each section between layers
        self.Ws = [np.random.randn(self.layer_sizes[l-1], self.layer_sizes[l]) / np.sqrt(self.layer_sizes[l-1])
                   for l in range(1, len(self.layer_sizes))]
        # Create a B for each layer except the last
        self.Bs = [np.zeros((1, self.layer_sizes[l+1])) for l in range(0, len(self.layer_sizes)-1)]


    def feedforward(self, X):
        next_input = X
        # Do first feed forward since it's different than the rest
        next_input = actFun(next_input.dot(self.Ws[0]) + self.Bs[0], 'tanh')
        for i in range(1, len(self.Ws)):
            next_input = self.layers[i-1].feedforward(next_input, self.Ws[i], self.Bs[i])
        # Compute the softmax function here
        exp_scores = np.exp(next_input)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

    def backprop(self, X, y, epsilon):
        num_examples = len(X)
        delta = self.probs
        delta[range(num_examples), y] -= 1
        reversed_layers = list(reversed(self.layers))
        reversed_Ws = list(reversed(self.Ws))
        reversed_Bs = list(reversed(self.Bs))
        # First backwards step must be done outside of the loop since it follows a different pattern
        reversed_layers[0].dLdZ = delta
        # print(reversed_layers[0].output.shape)
        # print(delta.shape)
        dW = 1./num_examples * reversed_layers[0].input.T.dot(delta)
        dB = delta.sum(axis=0)
        dW += self.reg_lambda * reversed_Ws[0]
        # Gradient descent parameter update
        reversed_Ws[0] += -epsilon * dW
        reversed_Bs[0] += -epsilon * dB

        # Now, for the rest of the weights and biases
        for l, i in zip(reversed_layers[1:], range(1, len(reversed_layers))):
            dW, dB = l.backprop(reversed_layers[i-1].dLdZ, reversed_Ws[i-1])
            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW += self.reg_lambda * reversed_Ws[i]

            # Gradient descent parameter update
            reversed_Ws[i] += -epsilon * dW
            reversed_Bs[i] += -epsilon * dB

        self.Ws = list(reversed(reversed_Ws))
        self.Bs = list(reversed(reversed_Bs))
        return None


    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        # print(self.probs.shape)
        # print(y.shape)
        data_loss = -np.sum(np.log(self.probs[range(num_examples), y]))

        # Add regulatization term to loss (optional)
        # data_loss += self.reg_lambda / 2. * (np.sum(np.square(self.Ws)) + np.sum(np.square(self.Ws)))
        return (1. / num_examples) * data_loss


    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            self.backprop(X, y, epsilon)
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print(
                    "Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


class Layer():
    def __init__(self, num_hidden, act_func_type):
        """
        :param num_hidden: Number of hidden units in this layer
        :param act_func_type: type of activator for this layer
        """
        self.num_hidden = num_hidden
        self.act_func_type = act_func_type
        self.input = None
        self.output = None
        self.z = None
        self.dLdZ = None

    def feedforward(self, input_act, weights, bias):
        """Apply the activation function """
        # print(input_act.shape)
        # print(weights.shape)
        # print(bias.shape)
        self.input = input_act
        self.z = input_act.dot(weights) + bias
        # print(self.z.shape)
        # print(self.input.shape)
        self.output = actFun(self.z, self.act_func_type)
        return self.output

    def backprop(self, dLdZ_right, W_right):
        """
        :param dLdZ_right: Derivative of loss with respect to the layer to the right's (layer + 1) net output
        :param W_right: The weight vector of the layer to the right (layer + 1)
        :return:
            dW: Derivative of loss with respect to input weights
            dB: Derivative of loss with respect to bias
        """
        # print(self.z.shape)
        # print(dLdZ_right.shape)
        # print(W_right.shape)
        # print(self.output.shape)
        self.dLdZ = np.multiply(diff_actFun(self.z, self.act_func_type), dLdZ_right.dot(W_right.T))
        dW = self.input.T.dot(self.dLdZ)
        dB = self.dLdZ.sum(axis=0)
        return dW, dB


def actFun(z, type):
    '''
    actFun computes the activation functions
    :param z: net input
    :param whichFun: tanh, sigmoid, or relu
    :return: activations
    '''

    # YOU IMPLMENT YOUR actFun HERE
    if type == 'tanh':
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    elif type == 'sigmoid':
        return 1. / (1. + np.exp(-z))
    elif type == 'relu':
        np.maximum(z, 0, z)
    else:
        raise RuntimeError(
            'Gave actFun an incorrect activation function name: ' + type)


def diff_actFun(z, type):
    '''
    diff_actFun computes the derivatives of the activation functions wrt the net input
    :param z: net input
    :param whichFun: Tanh, Sigmoid, or ReLU
    :return: the derivatives of the activation functions wrt the net input
    '''

    # YOU IMPLEMENT YOUR diff_actFun HERE
    if type == 'tanh':
        return 1. - ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))) ** 2
    elif type == 'sigmoid':
        return 1. / (1. + np.exp(-z)) * (1. - 1. / (1. + np.exp(-z)))
    elif type == 'relu':
        return 1. * (z > 0  )
    else:
        raise RuntimeError(
            'Gave diff_actFun an incorrect activation function name: ' + type)


def main():
    # # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, actFun_type='tanh')
    model = DeepNeuralNetwork(nn_input_dim=2, nn_output_dim=2, layers=[9, 10, 9],
                              layer_functions=['relu', 'tanh', 'tanh'])
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()