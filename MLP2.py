import random
import sys
import getopt
import time
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from colorama import init
from colorama import Fore


class MLP(object):
    def __init__(self, nbr_inputs, nbr_hidden, nbr_outputs, learning_rate, epochs):
        random.seed(1)
        init(autoreset=True)
        self.nbr_inputs = nbr_inputs + 1
        self.nbr_hidden = nbr_hidden
        self.nbr_outputs = nbr_outputs

        self.array_of_input = self.nbr_inputs * [1.0]
        self.array_of_hidden = self.nbr_hidden * [1.0]
        self.array_of_output = self.nbr_outputs * [1.0]

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.initialize_random_weight()

        self.change_inputs = np.zeros((self.nbr_inputs, self.nbr_hidden))
        self.change_output = np.zeros((self.nbr_hidden, self.nbr_outputs))
        self.print_mlp_struct()
        self.start_time = time.time()

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        return x * (1 - x)

    def print_mlp_struct(self):
        print(Fore.YELLOW + "########## MLP ##########")
        print(Fore.YELLOW + "Number of inputs neurons: " + str(self.nbr_inputs))
        print(Fore.YELLOW + "Number of hidden neurons: " + str(self.nbr_hidden))
        print(Fore.YELLOW + "Number of output neurons: " + str(self.nbr_outputs))
        print(Fore.YELLOW + "Learning rate: " + str(self.learning_rate))
        print(Fore.YELLOW + "Iterations: " + str(self.epochs))
        print(Fore.YELLOW + "#########################")

    def initialize_random_weight(self):
        self.input_hidden_weight = np.random.randn(self.nbr_inputs, self.nbr_hidden)
        self.hidden_output_weight = np.random.randn(self.nbr_hidden, self.nbr_outputs)

    def feed_forward(self, inputs):
        if len(inputs) == self.nbr_inputs - 1:
            for i in range(self.nbr_inputs - 1):
                self.array_of_input[i] = inputs[i]
            for j in range(self.nbr_hidden):
                sum = 0.0
                for i in range(self.nbr_inputs):
                    sum += self.array_of_input[i] * self.input_hidden_weight[i][j]
                self.array_of_hidden[j] = self.sigmoid(sum)
            for k in range(self.nbr_outputs):
                sum = 0.0
                for j in range(self.nbr_hidden):
                    sum += self.array_of_hidden[j] * self.hidden_output_weight[j][k]
                self.array_of_output[k] = self.sigmoid(sum)
            return self.array_of_output[:]
        else:
            raise ValueError("Incorrect number of inputs")

    def back_propagate(self, desired_outputs):
        if len(desired_outputs) == self.nbr_outputs:
            output_delta = [0.0] * self.nbr_outputs
            for k in range(self.nbr_outputs):
                error = -(desired_outputs[k] - self.array_of_output[k])
                output_delta[k] = self.d_sigmoid(self.array_of_output[k]) * error
            hidden_delta = [0.0] * self.nbr_hidden
            for j in range(self.nbr_hidden):
                for k in range(self.nbr_outputs):
                    change = output_delta[k] * self.array_of_hidden[j]
                    self.hidden_output_weight[j][k] -= self.learning_rate * change + self.change_output[j][k]
                    self.change_output[j][k] = change
            for i in range(self.nbr_inputs):
                for j in range(self.nbr_hidden):
                    change = hidden_delta[j] * self.array_of_input[i]
                    self.input_hidden_weight[i][j] -= self.learning_rate * change + self.change_inputs[i][j]
                    self.change_inputs[i][j] = change
            error = 0.0
            for k in range(len(desired_outputs)):
                error += 0.5 * (desired_outputs[k] - self.array_of_output[k]) ** 2
            return error
        else:
            raise ValueError("Incorrect number if desired outputs")

    def train(self, data):
        print(Fore.GREEN + "Training Started ...")
        error_per_epoch = []
        for i in range(self.epochs):
            error = 0.0
            bar = Bar('Processing', max=len(data))
            for d in data:
                inputs = d[0]
                desired_output = d[1]
                self.feed_forward(inputs)
                error = self.back_propagate(desired_output)
                bar.next()
            error_per_epoch.append(error)
            print(Fore.RED + " Epochs " + str(i) + " Global Error " + str(error))
            bar.finish()
        print(Fore.GREEN + "Training Finished in " + str(time.time() - self.start_time) + " sec")
        self.plot_errors(error_per_epoch)

    def plot_errors(self, errors):
        x = []
        for i in range(self.epochs):
            x.append(i)
        plt.plot(x, errors)
        plt.show()

    def predict(self, data):
        predictions = []
        for d in data:
            predictions.append(self.feed_forward(d))
        return predictions

    def load_my_dataset(self, file_path, _delimiter):
        dataset = np.loadtxt(file_path, delimiter=_delimiter)
        y = dataset[:,0:10]
        dataset = dataset[:,10:]
        dataset -= dataset.min()
        dataset /= dataset.max()
        out = []
        for i in range(dataset.shape[0]):
            exemples = list((dataset[i,:].tolist(), y[i].tolist()))
            out.append(exemples)
        return out

def main(argv):
    dataset_file = ''
    input = 0
    hidden = 0
    output = 0
    lrate = 0.0
    epochs = 0
    init(autoreset=True)
    try:
        opts, args = getopt.getopt(argv,"f:i:h:o:l:e:", ["help", "file=", "input=", "hidden=", "output=", "lrate=", "epochs="])
    except getopt.GetoptError:
        print(Fore.RED + "Argument error please refeer to following usage\n")
        usage()
    for opt, arg in opts:
        if opt == '--help':
            usage()
        elif opt in("-f", "--file"):
            dataset_file = arg
        elif opt in("-i", "--input"):
            input = int(arg)
        elif opt in("-h", "--hidden"):
            hidden = int(arg)
        elif opt in("-o", "--output"):
            output = int(arg)
        elif opt in("-l", "--lrate"):
            lrate = float(arg)
        elif opt in("-e", "--epochs"):
            epochs = int(arg)
        else:
            usage()

    mlp = MLP(input, hidden, output, lrate, epochs)
    X = mlp.load_my_dataset(dataset_file, ",")
    mlp.train(X)

def usage():
    print(Fore.RED + "##### MLP USAGE #####")
    print(Fore.YELLOW + "--- Arguments:")
    print("-f, --file: dataset path")
    print("-i, --input: number of inputs")
    print("-h, --hidden: number of hidden")
    print("-o, --output: number of output")
    print("-l, --lrate: learning rate")
    print("-e, --epochs: number of iteration")
    print(Fore.YELLOW + "--- Exemples:")
    print("python MLP2.py -f dataset.csv -i 64 -h 100 -o 10 -lrate 0.01 -e 20000")
    print(Fore.RED + "####################")
    sys.exit()

if __name__ == '__main__':
    main(sys.argv[1:])