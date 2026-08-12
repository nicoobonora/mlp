import network1
import network2
import mnist_loader
import plotter

# create the network
net1 = network1.Network([784, 16, 16, 10])
net2 = network2.Network([784, 16, 16, 10])

# prepare (format) data to use
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# train the network
accuracy_array_net_1 = net1.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# with lmbda = 0.5 we obtain that weights decay scales witght by 0.99995
accuracy_array_net_2 = net2.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.5, lmbda=5.0, test_data=test_data)

plotter.plot_accuracy([accuracy_array_net_1, accuracy_array_net_2])