#  The goal of this network is to improve the previous one by including the following features:
#  V - Weight initialization will be done by a standard deviation of 1/sqrt(n_in) instead of a standard deviation of 1
#  V - Biases distribution will keep having a standard deviation of 1
#  V - We'll use cross-entropy cost function + sigmoid to fix the slow learning problem (avoiding the sp problem) for the
#    output layer
#  V - The slow learning problem for the hidden layers is "solved" by the first point (weight initialization)
#  V - Implementation of L2 regulariazation (weight decay) to help the network generalize and keep consistency across multiple
#    possible training sessions
#  - Bonus: artifically expand the data set

from math import sqrt
import numpy as np
import random

class Network(object):

    # Constructor (eg of sizes: [784, 16, 10])
    def __init__(self, sizes):
        # Create the structure
        self.num_of_layers = len(sizes)
        # Biases still keep a standard deviation of 1
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        # Create the initial weights
        self.weights = self.intialize_weights(sizes)

    # --------- weights initialization
    def intialize_weights(self, sizes):
        # randn = random normal (Gaussian) distribution
        # randn has a default standard deviation of 1, so we force a standard deviation of x by multpliyng by x
        # [:-1] means "take everything from the array EXCEPT last element"
        # [1:] means "everything EXCEPT last element
        initial_weights = [1/np.sqrt(x) * np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        return initial_weights

    # --------- sigmoid function
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    # --------- sigmoid derivative function
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # --------- feed a data sample a (input layer activations) and return the output layer's activation
    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            # activation of a layer, reassigned at each cycle
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    # --------- cross-entropy cost function derivative (the L2 term is directly
    # added when computing the SGD step) (directly simplified sigmoid_prime)
    def ce_cost_function_derivative(self, a, y):
        return (a - y)

    # --------- takes an input x, feedforwards it, compares to the output y using the cross-entropy cost function, and
    # returns the desired gradient change (dCo/dw and dCo/db) for the given sample
    def backprop(self, x, y):
        # defining gradients components for Co

        #np.zeros() creates an array of zeros (ora a matrix, depending on the shape passed)
        # dCo/db array (initially empty)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # dCo/dw array (initially empty)
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward the sample: calculate the activation of each layer neurons and store them
        current_activation = x
        # array of each layers'activations, initialized to x only
        activations = [x]
        zs = [0] # the input layer doesnt have zs but directly activation, however i consider it 0 just
        # to be consistent with indexing in the backprop algorithm

        # compute each layers activations
        for w,b in zip(self.weights, self.biases):

            # store the zs of each layer neurons
            z = np.dot(w, current_activation) + b
            zs.append(z)
            
            # store the current actiovation of each layers neurons
            current_activation = self.sigmoid(z)
            activations.append(current_activation)
  
        # start the backprop algorithm
        delta = self.ce_cost_function_derivative(activations[-1], y)
        nabla_b[-1] = delta #dCo/db for the output layer
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #dCo/dw for the output layer

        # from layer l-1 to the first hidden layer
        for l in range(2,self.num_of_layers):
            # so for three layers l will be 1, 2
            z = zs[-l] # z requires this indexing because the z array is smaller than the a array
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.sigmoid_prime(z)
            nabla_b[-l] = delta #dCo/db
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #dCo/dw

        return [nabla_b, nabla_w]


    # --------- take a mini batch of data -> backprop for each data sample -> compute a gradient step
    # (aka, update weights and biases).
    # Note: lambda is a py keyword, so let's just call it lmbda
    # n is the full training_data set length, not the mini batch length.
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        
        # setup for the whole gradient descent array, as done in backprop
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        for sample, label in mini_batch:

            # backprop a sample
            delta_nabla_b, delta_nabla_w = self.backprop(sample, label)

            # sum requested gradient step of each sample to the existing ones
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        
        

        # Here we apply weight decay (L2 regularization) to the step
        # w == old weights set, nw = new weight set
        # len(mini_batch) is to finalize the mean of the costs of the single samples produced in the loop above
        self.weights = [(1-(eta*lmbda)/n)*w - eta*nw/(len(mini_batch)) for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta*nb/(len(mini_batch)) for b,nb in zip(self.biases, nabla_b)]

    # --------- SGD
    def SGD(self, training_data, eta, lmbda, epochs, mini_batch_size, validation_data=None, test_data=None):

        # the n in the weight-decay-adjusted formula for gradient step is
        # the total length of the training_data set 
        n = len(training_data)

        # So take a mini batch -> backprop all its data -> compute gradient step and apply it (change weights and biases)
        # -> do this for all the data -> that's one epoch
        # -> repeat this for the requested epochs
        for epoch in range(epochs):
            
            # Randomize data
            random.shuffle(training_data)

            # build mini_batches for the epoch
            mini_batches = [
                training_data[k: k+mini_batch_size]
                # parameters are start, stop, range. so starts from 0, creates a sub-array with elements from 0 to 10-1 (or whatever
                # the desired mini_batch size is), takes a step of 10, creates a sub-array with elements from 10 to 20-1, and so on
                # for the whole training dataset
                for k in range(0, n, mini_batch_size)
            ]
        
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            # accuracy array
            accuracy_array = []
            if test_data:
                # Syntax for string nesting
                evaluation = self.evaluate(test_data)
                accuracy_array.append(evaluation)
                print("Epoch {0} : {1} / {2}".format(
                    epoch, evaluation, len(test_data)
                ))
            else:
                print("Epoch {0} complete".format(j))
            
        print("----- Net 1 Finished -----")
        return accuracy_array


    # --------- evaluate an epoch
    def evaluate(self, test_data):
        # argmax "returns the indices of the maximum values along an axis". In this case, the axis is the axis of the desired result,
        # so for example [0 0 1.0 0 0 0 0 0 0 0 0] if the desired result is the number 2, and the index
        # which selects the value is the output layer's activation array,
        # so the output results, returned by feedforward. so argmax gets the index of the highest activation of the output layer, 
        # and picks the corresponding desired output layer value. for example, if y is [0 0 1.0 0 0 0 0 0 0 0 0] and the output layer
        # gives something like [0.1 0.02 0.9 0.3 0.01 0.01 0.08 0.07 0.1 0.2], it means the network predicted with a certainty of 0.9
        # that the input is a 2 (third value of the output layer), so the third value of the y array is picked (corresponding to the
        # number 2).

        # so test_results in an array (guessed result, desired result) for each given sample
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        
        # if x == y (prediction correct) it produces "1" and sums it, else 0. The final result is the number of right predictions
        return sum(int(x == y) for (x, y) in test_results)