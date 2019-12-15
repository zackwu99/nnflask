import numpy as np
from lib import activation_function as af
from lib import data_reader as rd
from lib import data_parser as pr
from lib import weights_writer as wr
from lib import weights_reader as wrd

class NeuralNetwork:
    def __init__(self, input_size: int, num_hidden_layer: int, hidden_size: int, output_size: int,
                 learning_rate: float):
        self.input_size = input_size
        self.num_hidden_layer = num_hidden_layer
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_weights = np.zeros((hidden_size, input_size))
        self.output_weights = np.zeros((output_size, hidden_size))
        self.hidden_weights = np.zeros((num_hidden_layer - 1, hidden_size * hidden_size))
        self.learning_rate = learning_rate
        self.input_layer_learning_rates = np.full((self.hidden_size, self.input_size), learning_rate)
        self.output_layer_learning_rates = np.full((self.output_size, self.hidden_size), learning_rate)
        self.hidden_layer_learning_rates = np.full((self.hidden_size, self.hidden_size), learning_rate)
        self.repeat_training = 1

    # =================================================================================================================
    # Public functions
    # =================================================================================================================
    def train(self, data_file: str):
        print('START TRAINING WITH RANDOM WEIGHTS')

        # Read data from file
        training_data = rd.read_file(data_file)

        # Repeat training with same data sets
        for l in range(self.repeat_training):
            self.run_training(training_data)

        print('END OF TRAINING')

    def run_training(self, training_data):
        # Parse training data
        number_of_training_row: int = len(training_data)

        for iteration in range(number_of_training_row):
            print('Iteration =', iteration)

            # Get input row at iteration
            img_row_data: str = training_data[iteration]

            # Retrieve target for this training data row
            targets = pr.parse_target(img_row_data, self.output_size)

            # Retrieve training data row
            input_data = pr.parse_scale_image_data(img_row_data)

            self.train_one_row(input_data, targets)

    def train_one_row(self, input_data, targets):
        # Convert input to a vector with shape (784,1)
        x = np.array([input_data]).T

        # Calculate inputs to first hidden layer
        # i has a shape of (hidden_size,1)
        i = np.dot(self.input_weights, x)

        gradient_input = af.derivative_sigmoid(i) * self.input_layer_learning_rates * x.T

        # Keep gradients for all hidden layers
        gradients_hidden = []

        # Calculate output of first hidden layer
        # h is the output of hidden layer, it has shape (hidden_size,1), same as i
        h = af.sigmoid(i)

        # Assign hh to h. Because hh will change in loop, and we want to keep h value unchanged
        # The first hidden layer has been processed above
        # so the loop will start to process from second hidden layer
        hh = h
        # Loop over each hidden layer
        for j in range(self.num_hidden_layer - 1):
            # ww is weights for hidden layer j and it has shape of (hidden_size,hidden_size)
            ww = self.hidden_weights[j].reshape((self.hidden_size, self.hidden_size))

            # ii is the input to next hidden layer and it has shape of (hidden_size,1)
            ii = np.dot(ww, hh)

            # we can calculate part of gradients without error
            # The shapes in below calculation are (hidden_size,1) (hidden_size, hidden_size), (1, hidden_size)
            # gg has shape of (hidden_size, hidden_size)
            gg = af.derivative_sigmoid(ii) * self.hidden_layer_learning_rates * hh.T
            gradients_hidden.append(gg)

            # hh can calculate part of gradients without error
            hh = af.sigmoid(ii)

        # Calculate inputs to output layer
        i = np.dot(self.output_weights, hh)

        # Calculate output of output layer
        y = af.sigmoid(i)

        # Calculate error of output layer
        t = np.array([targets]).T
        # e = 0.5 * (t - y) * (t - y)
        e = (t - y)

        # back_propagation
        # Calculate gradient from output layer
        delta_w = -e * af.derivative_sigmoid(i) * self.output_layer_learning_rates * hh.T

        # Caculate the error from last hidden layer
        ee = np.dot(np.transpose(self.output_weights), e)

        # Update output layer weights
        self.output_weights -= delta_w

        # Loop backwards for hidden layer
        # The number of hidden layer weights =  num hidden layer - 1
        for j in range(self.num_hidden_layer - 1, 0, -1):
            # Hidden layer weight index k = j - 1
            k = j - 1

            # Get gradients from gradients list
            gg = gradients_hidden[k]

            # Calculate the delta of weights from error and gradient
            delta_ww = -ee * gg

            # Get hidden weights
            ww = self.hidden_weights[k].reshape((self.hidden_size, self.hidden_size))

            # Calculate the error for previous hidden layer before update the weigts
            ee = np.dot(np.transpose(ww), ee)

            # Update the weights for hidden layer
            ww -= delta_ww

            # Convert ww and update hidden weights k-th row
            tw = np.array(ww.reshape((1, self.hidden_size * self.hidden_size)))
            self.hidden_weights[k, :] = tw

        # Calculate delta of weights for input layer at last
        delta_w = -ee * gradient_input
        self.input_weights -= delta_w

    def load_weights(self, input_layer_weight_file: str, output_layer_weight_file: str, hidden_layer_weight_file: str):
        print('LOADING WEIGHTS FROM FILES')
        self.input_weights = wrd.read_weights(input_layer_weight_file)
        self.output_weights = wrd.read_weights(output_layer_weight_file)
        self.hidden_weights = wrd.read_weights(hidden_layer_weight_file)
        print('DONE LOADING WEIGHTS FROM FILES')

    def save_weights(self, input_layer_weight_file: str, output_layer_weight_file: str, hidden_layer_weight_file: str):
        print('SAVING WEIGHTS TO FILES')
        wr.write_file(input_layer_weight_file, self.input_weights)
        wr.write_file(output_layer_weight_file, self.output_weights)
        wr.write_file(hidden_layer_weight_file, self.hidden_weights)

    def test(self, test_file: str):
        print('START RUNNING TEST FILE')
        test_data = rd.read_file(test_file)

        # Parse test data
        number_of_test_row: int = len(test_data)
        correct_counnt: int = 0
        for iteration in range(number_of_test_row):
            # Get input row at iteration
            row_data: str = test_data[iteration]

            # image_data = pr.parse_image_data(row_data)
            # matplotlib.pyplot.imshow(image_data, cmap='Greys', interpolation='None')

            (probability, guess_number) = self.test_one_row(row_data)
            t = pr.parse_target_number(row_data)
            print("Iteration: % 2d. Highest Probability= % 5.3f" % (iteration, probability * 100))
            if t == guess_number:
                print('GUESS CORRECTLY. The number is ', guess_number)
                correct_counnt += 1
            else:
                print('GUESS WRONG. The number is not ', guess_number)

        percentage: float = correct_counnt / number_of_test_row
        print("Correct Percentage = %", percentage * 100)
        print('END RUNNING TEST FILE')

    def test_one_row(self, row_data):
        # Retrieve training data row
        input_data = pr.parse_scale_image_data(row_data)
        x = np.array([input_data]).T

        # Calculate inputs to 1st hidden layer
        i = np.dot(self.input_weights, x)

        # Calculate output of hidden layer
        h = af.sigmoid(i)

        hh = h
        # Loop over each hidden layer
        for j in range(self.num_hidden_layer - 1):
            # ww is weights for hidden layer j and it has shape of (hidden_size,hidden_size)
            ww = self.hidden_weights[j].reshape((self.hidden_size, self.hidden_size))

            # ii is the input to next hidden layer and it has shape of (hidden_size,1)
            ii = np.dot(ww, hh)

            # hh can calculate part of gradients without error
            hh = af.sigmoid(ii)

        # Calculate inputs to output layer
        i = np.dot(self.output_weights, hh)

        # Calculate output of output layer
        y = af.sigmoid(i)

        number_and_index = max([(max_value, index) for index, max_value in enumerate(y)])
        probability = number_and_index[0]
        guess_number = number_and_index[1]

        return probability, guess_number

    def check_one_row(self, row_data):
        # Retrieve training data row
        input_data = pr.parse_scale_image_data_without_target(row_data)
        x = np.array([input_data]).T

        # Calculate inputs to 1st hidden layer
        i = np.dot(self.input_weights, x)

        # Calculate output of hidden layer
        h = af.sigmoid(i)

        hh = h
        # Loop over each hidden layer
        for j in range(self.num_hidden_layer - 1):
            # ww is weights for hidden layer j and it has shape of (hidden_size,hidden_size)
            ww = self.hidden_weights[j].reshape((self.hidden_size, self.hidden_size))

            # ii is the input to next hidden layer and it has shape of (hidden_size,1)
            ii = np.dot(ww, hh)

            # hh can calculate part of gradients without error
            hh = af.sigmoid(ii)

        # Calculate inputs to output layer
        i = np.dot(self.output_weights, hh)

        # Calculate output of output layer
        y = af.sigmoid(i)

        number_and_index = max([(max_value, index) for index, max_value in enumerate(y)])
        probability = number_and_index[0]
        guess_number = number_and_index[1]

        return probability, guess_number

    def initialize_random_weights(self):
        # Set weights for input layer
        self.input_weights = 2 * np.random.rand(self.hidden_size, self.input_size) - 1

        # Set weights for hidden layers
        for i in range(self.num_hidden_layer - 1):
            w = 2 * np.random.rand(self.hidden_size, self.hidden_size) - 1
            self.hidden_weights[i, :] = w.reshape((1, self.hidden_size * self.hidden_size))

        # Set weights for out layer
        self.output_weights = 2 * np.random.rand(self.output_size, self.hidden_size) - 1
