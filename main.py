import numpy as np

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
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z)) # if z is a vector, np.exp automatically applies sigmoid to each element

    # ------------

    # Calculates the activation for each neuron ofthe network
    # a is the vector of one layers' activations. e.g, the first a given to the function will be an array of 784 activations
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
            a = sigmoid(np.dot(w, a) + b)
        return a

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
    # - data is the data itself (a list of data to test)
    # - test_data, when provided (optional), will be evaluated against the network prediction and partial progress will
    #   be printed out. Useful for tracking progress, but very slow
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, data, test_data=None):
    
        # Defining lengths of datas
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        # range(epochs) is an iterator with *epochs* "elements" (e.g. epochs = 30 -> the loop will iterate 30 times)
        # this is the code that gets called for each epoch
        for j in range(epochs):
            # Random shuffle data
            np.random.shuffle(training_data)

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
            if test_data:
                # Syntax for string nesting
                print("Epoch {0} : {1} / {2}".format(
                    j, self.evaluate(test_data), n_test
                ))
            else:
                print("Epoch {0} complete".format(j))


net = Network([784, 16, 16, 10])
