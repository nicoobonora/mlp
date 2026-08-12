import numpy as np
import random

# NN recognizing digits

class Network(object):

    # ------------

    # Constructor
    # sizes: array of number of neurons per layer, like [784, 16, 16, 10]
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Randomly generate biases
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        # sizes[:-1] grabs every items of sizes excepts the last, so [784, 16, 16]
        # sizes[1:] grabs every items except the first, so [16, 16, 10]
        # zips pairs two sets into an iteretor of touples, so zip() will be [(784, 16), (16,16), (16,10)]
        # np.random.randn(y,x) loops through that, so weights will be an array of matrixes of dimensions [16x784], [16x16], [10x16]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    # ------------

    # As in the notes, z is the weighted sum, (wa + b)
    # Sigmoid is the function to "normalize" the weighted sum between 0 and 1
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z)) # if z is a vector, np.exp automatically applies sigmoid to each element

    # ------------

    # Calculates the activation for each neuron ofthe network
    # a is the vector of one layers' activations. e.g, the first a given to the function will be an array of 784 activations
    
    # this function doesnt change ANY weights or bias but just inputs an input a and returns the output produced by the network
    # with the current weights and biases
    def feedforward(self, a):
        # zip(self.biases, self.weights) returns an iterator where each element consists of the biases of a layer
        # and the corresponding weight matrixes. e.g. the first element will be [(*array of 16 biases*, *[16x784] matrix*)]
        for b,w in zip(self.biases, self.weights):
            # Definition of a layer's activation
            # Here all w, b and a are vector. So for the first iteration, a will be the 784 activations of the first layer,
            # w the 16x784 weights matrix, and b a 16 elements vector (the biases of the second layer). This function will of course be called after recalculating weights.
            # dot is the dot product, or in this case a matrix x vector product.
            # notice this isnt exactly a reassignment. the a input is the 784 activations (a(0)), these one never change (they represent
            # the handwritten digit). So on the first loop, w*a + b returns a 16 rows vector, computed whit the new weight (so it computes)
            # a(1). On the next loop, w*a + b returns another 16 rows vector, a(2). On the third and last iteration, returns a 10 rows vector (the output layer).
            # In the end of the function, the returned a is just the output layer computed with the new weights, which is the only thing
            # we really care about. Notice network isnt updated, we are only calculating, feed forward, the output layer's activation
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    # --------------

    # Code for the backpropagation algorithm
    # - x is the input, a 748 vector, representing a data sample
    # - y is the label, aka the desired output, so a 10 vector
    # The scope of the backprop is to return the partial derivatives of the cost of the passed sample (So C0 for the first sample,
    # C1 for the second, and so on) with respect to biases and weights, for ALL the layers

    def backprop(self, x, y):
        
        # Define the arrays that will be the requested gradient change by this data sample (initialized to an array of zeros)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x # the input (the data sample)
        activations = [x] # list to store each layer activations, so this will have 4 elements
        zs = [] # list to store all the z vectors, meaning the weighted sum of each neuron of each layer
        # we need to keep zs and activations separate because we need to take partial derivates with respect to each

        # Loops through biases and wights arrays of each layer 
        for b,w in zip(self.biases, self.weights):
            # - First loop:
            #   input (784) x weights[0] ([16x784]) + b -> array of 16 elements with each containing the weighted sum
            #   append that to zs (second layer's weighted sums)
            #   sigmoid(z) gives an array of size 16 with the new activations
            #   append that to activations
            # - Second loop:
            #   second layer (16) + weights[1] ([16x16]) -> array of 16 elements containing the weighted sums
            #   ...
            # - Third layer:
            #   same things ....  
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        # Finished the feedforward phase, so given the sample and filled all the activations (and zs),
        # we can start applying backpropagation
        
        # - delta is the diff between the output layer and the desired output
        # - activations[-1] is the last layer of neurons (the output)
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1]) # 
        # Fill the temporary "gradient" for this data sample (so the gradient with partial derivative with C0) with 
        # the partuial derivatives with respect to the output layer biases and weights
        nabla_b[-1] = delta # dCo/db
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # dCo/dw, activation[-2] refers to the third layer's activations

        # Filling of the previous layers

        # Even tho in the notes L was used to represente the last layer, here we use it the opposite way to take advantage of the
        # python negative indexes, so here L=1 means the last layer, L=2 the third, ...
        # Here we start from the second layer
        for l in range(2, self.num_layers):
            z = zs[-l] # for the first loop, it's the third layer. So z is the array of all weighted sum of the -l layer
            sp = self.sigmoid_prime(z) # partial derivative of the z vector
            # the right delta is the delta (difference between the layers activations and the expected) of the layer after.
            # so what it's doing for the third layer is 
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta # dC0/db(L-1)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # dC0/dw(L-1)
        
        # finally returns the array of derivatives of C0 with respect to every weight and biases, then for the next sample it
        # will be the same for C1, then for C2, ...
        # the update_mini_batch() function will call backprop() for each sample of the minibatch, sum all the arrays of derivatives
        # calculated by backprop, do the mean, and then actually compute the new weights and biases. Repeat this for every mini_batch
        # and you got a trained network
        return (nabla_b, nabla_w)

    # --------------

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

    # --------------

    # given an output and the desired output, returns the difference. theoritically, it should be 2*(output_activation - y),
    # but as described in the notes 2 is not that important, as we are going to toggle directly eta, which would have been multiplied
    # by 2
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    # --------------
    
    # This is the derivative of the sigmoid function
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    # --------------

    # Called in the SGD function, this updates the network's weights and biases by applying
    # gradient descent using backpropagation to a single mini batch

    # - mini_batch is a list of touples (x,y), where x is the input (a 784 vector encoding the handwritten digit) and y is the
    #.  expected result (a 10 vector encoding the result (the label), eg [0 0 0 1.0 0 0 0 0 0 0] for the number 3)
    def update_mini_batch(self, mini_batch, eta):
        # Remembering nabla is the symbol of the gradient (and rotor and divergence), nabla_b and nabla_w will be filled
        # with the partial derivatives of the cost with respect to the biases and weights, which will be computed via backpropagation
        # For now, we initialize them to empty arrays of 0 (meaning "no step will have to be made")
        # note: shape just gives the shape of an array/matrix
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # For each data sample, we calculate the partial derivative of its cost with respect to each weight (via backpropagation),
        # sum them all up in the loop (collecting the sum in nabla_b and nabla_w), and fo the mean after to calculate the average required
        # wights change and bias change
        for x,y in mini_batch:
            # Core of the whole network, backprop gives us all the partial derivatives for one data sample, and we assign them to delta_nabla_b and
            # delta_nabla_w
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)

            # nb, dnb, nw, dnw are just abbreviationts for nabla_..., delta_nabla_...
            # Essentialy, at each iteration of the for loop, so for each data sample, we sum the required change of that data sample to 
            # the final nabla, and do the mean. This is exactly the process described in the table at page 12 of the notes.
            # so for example, for the weight 1 the first data sample requires +0.18, then for the second data sample weight 1 requires -0.37, ...,
            # at the end the requested step from weight 1, so the partial derivative of the cost with respect to weight 1, will be the sum 
            # of all the requestests from each data sample, divided by the number of samples in the mini_batch. 
            # Note: we do the mean after. For now, in the loop, we just collect the sums.
            # This of course is done for each weight and each bias.

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # /len(mini_batch) is the mean: nw contains an array of sums of the partial derivative
        # for each data sample, by multipling by 1/len(mini_batch) we divide each element of the array of partial derivatives (nablas) by the number of data
        # samples used. With this, we are updating the network with the new weights and biases required by this mini_batch
        self.weights = [w - (eta * 1/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta * 1/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    # --------------

    # Stochastic gradient descent
    # As in the notes, this is the algorithm to call the gradient operation more effeciently, computing it not with all the
    # data at once but rather iterating through small pieces of the dataset at a time (minibatches) 

    # Parameters:
    # - training_data is a list of touples (x, y) where x is the training input (a 784 vector representing the digit) and
    #   y is the expected output (a 10 vector representing the label, eg [0 0 0 1.0 0 0 0 0 0 0] for the number 3)
    # - epochs is a number representing how many times we want to loop through the entire dataset
    # - eta (same greek letter as the rendimento) represents the learning rate
    # - mini_batch_size is of course the size of a mini batch, so the size of a subset of the data after which we want to
    #   evaluate the gradient
    # - test_data, when provided (optional), will be evaluated against the network prediction and partial progress will
    #   be printed out. Useful for tracking progress, but very slow
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    
        # Defining lengths of datas
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        # range(epochs) is an iterator with *epochs* "elements" (e.g. epochs = 30 -> the loop will iterate 30 times)
        # this is the code that gets called for each epoch
        for j in range(epochs):
            # Random shuffle data
            random.shuffle(training_data)

            # So if we want mini_batches to be of 10 elements, mini_batches will be an arrays of sub-arrays,
            # each containing an array of 10 elements (10 (x,y) touples)
            mini_batches = [
                training_data[k: k+mini_batch_size]
                # parameters are start, stop, range. so starts from 0, creates a sub-array with elements from 0 to 10-1 (or whatever
                # the desired mini_batch size is), takes a step of 10, creates a sub-array with elements from 10 to 20-1, and so on
                # for the whole training dataset
                for k in range(0, n, mini_batch_size)
            ]

            # For each mini_batch, we compute and apply a step of gradient descent, done by update_mini_batch(), using
            # just the data from that mini batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Logging progress
            # essentialy, each epoch, we first train the net (above code) and then try to feed a sample data (without computing
            # backpropagation after, just evaluation) and check how the net progressed
            # accuracy array
            accuracy_array = []
            if test_data:
                # Syntax for string nesting
                evaluation = self.evaluate(test_data)
                accuracy_array.append(evaluation)
                print("Epoch {0} : {1} / {2}".format(
                    j, evaluation, n_test
                ))
            else:
                print("Epoch {0} complete".format(j))

        print("----- Net 1 Finished -----")    
        return accuracy_array