
import numpy as np
from pickle import load, dump
from os.path import exists

class NeuronalNetwork:
  def __init__(self, input_n, hidden_n, output_n, from_file = False):
    self.file = 'training/training.bin'
    if from_file == False:
      self.weights_ih = np.random.normal(0, 1 / np.sqrt(hidden_n), (hidden_n, input_n))     
      self.weights_ho = np.random.normal(0, 1 / np.sqrt(output_n), (output_n, hidden_n))
      print('Network initialised with normal distributed weights.')
    else:
      if exists(self.file):
        with open(self.file, 'rb') as fd:
          print('Weights were found in the training file.')
          weights = load(fd)
          self.weights_ih = weights[0]
          self.weights_ho = weights[1]
      else:
        print('Training file not found in the training folder.')

  def sigmoid(self, x):
    return 1 / (np.exp(-x) + 1)

  # - Method used to feed forward the network. can also be used for testing the network.
  # - Calculate the activation cost for each layer of neuron given the previous layr
  # and the matrix of weights between those two layers.
  def feed_forward(self, inputs):
    layers = []
    layers.append(np.array(inputs, ndmin = 2).T)
    layers.append(self.sigmoid(np.dot(self.weights_ih, layers[0])))
    layers.append(self.sigmoid(np.dot(self.weights_ho, layers[1])))
    return layers
  
  # Method to train the netork.
  # inputs and targets - vectors in pair to do the training
  # rate - learning rate used in the backpropagation(Alpha in the above formula)
  def train_network(self, inputs, targets, rate):
    # - Error calculated between the output layer and the targets(labels) 'layer' is simply
    # output - target. This will then be backpropagated to the previous hidden layers to be used
    # in the gradient descent algorithm to adjust the weights matrices
    layers = self.feed_forward(inputs)
    errors = []
    errors.append(layers[2] - targets)
    errors.append(np.dot(self.weights_ho.T, errors[0]))
    # Apllying gradient descent
    self.weights_ho -= rate * np.dot((errors[0] * layers[2] * (1 - layers[2])), layers[1].T)
    self.weights_ih -= rate * np.dot((errors[1] * layers[1] * (1 - layers[1])), layers[0].T)
    
  def prepare_set_from_str(self, string):
    data = string.split(',')
    data_f_array = np.asfarray(data, dtype = 'float')
    label = int(data_f_array[0])
    label_array = np.zeros((10, 1)) + .01 
    label_array[label][0] = .998
    data_f_array = data_f_array[1:]
    return {'label': label, 'target': label_array, 'training_set': data_f_array / 255 * .99 + 0.01,}
  
  def train_from_file(self, rate, path, limit = 100, write = False):
    print(f'Training with {limit} sets.')
    with open(path) as fd:
      line = fd.readline()
      count = 1
      while line and count <= limit:
        training_set = self.prepare_set_from_str(line)
        self.train_network(training_set['training_set'], training_set['target'], rate)
        line = fd.readline()
        count += 1
    if write:
      with open(self.file, 'wb') as fd:
        weights = [self.weights_ih, self.weights_ho]
        dump(weights, fd)
        print('Weights matrices writen to training file in the training folder.')
        
  def test_from_file(self, path, limit = 100):
    score = 0
    count = 1
    with open(path) as fd:
      line = fd.readline()
      while line and count <= limit:
        test = self.prepare_set_from_str(line)
        output = self.feed_forward(test['training_set'])[-1]
        result = np.argmax(output)
        if result == test['label']:
          score += 1
        #print(f'Test number: {count}.')
        #print('Expected output:', test['label'])
        #print('Output:', result, '\n')
        line = fd.readline()
        count += 1
    count -= 1
    accuracy = score * 100 / count
    print(f'Accuracy after {count} tests: {accuracy}%')
    return accuracy

net = NeuronalNetwork(784, 160, 10, from_file = True)
#net.train_from_file(.3, path = 'csv_sets/mnist_train.csv', limit = 55000, write = True)
net.test_from_file(path = 'csv_sets/mnist_test.csv', limit = 10000)