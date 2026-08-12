import matplotlib.pyplot as plt

# not working, to understand why

def plot_accuracy(arrays):
    for array in arrays:
        plt.plot(array)
        plt.show()