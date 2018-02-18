# MNIST-neuronal-net
Writing a neuronal network to learn to recognize digits from the MNIST database.

The database provides 60.000 training sets and 10.000 test sets.

With a hidden layer of 300 nodes, learning rate of 0.2, 5 epochs and 
weights initialised from a normal distribution I managed to get to a 
97.1% accuracy.

Accuracy updates:

- 1 epoch, 0.33 learning rate, 160 nodes hidden layer, 55.000 training
inputs => 95.14% accuracy

![95](https://github.com/MarcusGitAccount/MNIST-neuronal-net/blob/master/accuracy/accuracy.png)

- 5 epochs, 0.22 learning rate, 300 node hidden layer, 60.000 training inputs
-> 97.1% accuracy

![97](https://github.com/MarcusGitAccount/MNIST-neuronal-net/blob/master/accuracy/accuracy_2.png)

Data used from the following source:
https://pjreddie.com/projects/mnist-in-csv/

Python packages used:
- pickle
- numpy

TODO:
- Learn to use CUDA and rewrite all the code to work on the GPU.

License: GPLV3
