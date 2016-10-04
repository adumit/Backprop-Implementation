__author__ = 'andrew_dumit'
import numpy as np
from three_layer_neural_network import actFun, diff_actFun, NeuralNetwork, generate_data


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
        self.dWs = []
        self.dBs = []


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
        data_loss = -np.sum(np.log(self.probs[range(num_examples), y]))

        # Add regulatization term to loss (optional)
        sum_of_squares = 0
        for w in self.Ws:
            sum_of_squares += np.sum(np.square(w))
        data_loss += self.reg_lambda / 2. * sum_of_squares
        return (1. / num_examples) * data_loss

    def calculate_misclassifcation(self, X, y):
        num_examples = len(X)
        self.feedforward(X)

        # Calculate misclassification
        # cor_nums = np.apply_along_axis(lambda a: np.argmax(a), 1, y)
        cor_nums = y
        pred_nums = np.apply_along_axis(lambda a: np.argmax(a), 1, self.probs)
        return 1. - np.sum(cor_nums == pred_nums)/num_examples

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            self.feedforward(X)
            self.backprop(X, y, epsilon)
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
                print("Misclassification after iteration %i: %f" % (i, self.calculate_misclassifcation(X, y)))


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
        self.input = input_act
        self.z = input_act.dot(weights) + bias
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
        self.dLdZ = np.multiply(diff_actFun(self.z, self.act_func_type), dLdZ_right.dot(W_right.T))
        dW = self.input.T.dot(self.dLdZ)
        dB = self.dLdZ.sum(axis=0)
        return dW, dB


def get_mnist_data_batch(mnist_data):
    batch = mnist_data.train.next_batch(3000)
    data = batch[0]
    labels = batch[1]
    return data,labels.astype(int)


def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data')
    # Generate make moons data
    # X, y = generate_data()
    # Generate MNIST data
    X, y = get_mnist_data_batch(mnist)
    # model = DeepNeuralNetwork(nn_input_dim=2, nn_output_dim=2, layers=[500, 300],
    #                           layer_functions=['tanh', 'tanh'], reg_lambda=0)
    model = DeepNeuralNetwork(nn_input_dim=784, nn_output_dim=10, layers=[500, 300],
                              layer_functions=['tanh', 'tanh'], reg_lambda=0.0001)
    model.fit_model(X,y, num_passes=20000, epsilon=0.0001)
    model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()