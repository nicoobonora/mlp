import network
import mnist_loader

# create the network
net = network.Network([784, 16, 16, 10])

# prepare (format) data to use
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# train the network
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)