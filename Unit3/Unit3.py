#################################################
# Examples and Exercises of Unit 3 
##################################################
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from IPython import embed

class NeuralNetworkUnit:
    def __init__(self, num_inputs, activation_function):
        self.weights = np.random.rand(num_inputs, 1)
        self.bias = np.random.rand()
        self.activation_function = activation_function
    
    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(z)


def ReLU(z):
  return np.maximum(0, z)

def hyp_tan(z):
  return np.tanh(z)

def z(x, weights, bias):
  return np.dot(x, weights) + bias

def unit_step(z):
  return np.where(z >=0, 1, 0)

def NeuralNetwork(inputs, weights, bias, activation_function):
  z_ = z(inputs, weights, bias)
  return activation_function(z_)

def ActFunc_6(z):
  return 2*z - 3 

def NAND(x1, x2):
  return (not (x1 and x2))

# 4. Neural Network Units
if False:
  x = np.array([1, 0])
  w0 = -3
  w = np.array([1, -1])

  print("NeuralNetwork using ReLU: ", NeuralNetwork(x, w, w0, ReLU))
  print("NeuralNetwork using HypTan: ", NeuralNetwor(x, w, w0, hyp_tan))

# 5. Introduction to Deep Neural Networks
if False:
  # Define the input and expected output for the NAND function
  x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y = []
  for inputs in x:
    x1 = inputs[0]
    x2 = inputs[1]
    y.append([int(NAND(x1, x2))])

  # expected output for the NAND function
  y = np.array(y)


  # define the neural network with 2 inputs and 1 output
  nn_unit = NeuralNetworkUnit(2, unit_step)
  # try with random weights and bias such the output of a neural network with y = U(z) matches the NAND function
  learning_rate = 0.1
  iterations = 10000
  for i in range(iterations):
    # compute predicted output
    y_pred = nn_unit.forward(x)

    # compute tje error between the predicted and expected outputs
    error = y - y_pred

    # update weights and bias using gradient descent
    nn_unit.weights += learning_rate * np.dot(x.T, error)
    nn_unit.bias += learning_rate * np.sum(error)

  # Test the neural network on new inputs
  print("Results after training NeuralNetworkUnit: ")
  print("-----------------------------------------")
  test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  test_outputs = nn_unit.forward(test_inputs) 
  exp_outputs = []
  print("Expected Outputs: ")
  for i in range(len(test_inputs)):
    exp_outputs.append([int(NAND(test_inputs[i][0], test_inputs[i][1]))])
  print(exp_outputs)
  print("Predicted Outputs: ")
  print(test_outputs)
  print("Weights: ", nn_unit.weights)
  print("Bias: ", nn_unit.bias)


# 6. Hidden layers
if False:
  inputs = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
  preds = np.array([1, -1, -1, 1])
  weights_1 = np.array([[0, 0], [2, -2], [-2, -2]])
  weights_2 = np.array([[0, 0], [-2, -2], [2, 2]])
  bias = np.array([[0], [1], [1]])

  fig = plt.figure()
  for i in range(len(inputs)):
    color = 'blue' if preds[i] == 1 else 'red'
    plt.plot(inputs[i][0], inputs[i][1], ls='None', marker='.', color=color)

  plt.xlabel('x1')
  plt.ylabel('x2')
  
  
  for j in range(len(bias)):
    fig = plt.figure()
    for i in range(len(inputs)):
      f1 = ActFunc_6(z(inputs[i], weights_1[j], bias[j]))
      f2 = ActFunc_6(z(inputs[i], weights_2[j], bias[j]))
      #result.append([f1, f2])
      #colors.append(preds[i])
      c = 'blue' if preds[i] == 1 else 'red'
      plt.plot(f1, f2, ls='None', marker='.', color=c)

    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Answer %i' %j)

  # CODE BELOW IS NOT WELL IMPLEMENTED
  if False:
    # Non-linear Activation Functions
    # consider now the weights:
    weights_1 = np.array([1, -1])
    weights_2 = np.array([-1, -1])
    bias = np.array([1, 1])

    results = defaultdict(list)
    function = ['5z-1', 'ReLU', 'tanh', 'z']
    result_ = []
    
    for i in range(len(inputs)):
      
      f1_ = 5*z(inputs[i], weights_1[0], bias[0]) - 2
      f2_ = 5*z(inputs[i], weights_2[1], bias[1]) - 2
      
      results['5z-1'].append([f1_, f2_])

      f1_ReLU = ReLU(z(inputs[i], weights_1[0], bias[0]))
      f2_ReLU = ReLU(z(inputs[i], weights_2[1], bias[1]))

      results['ReLU'].append([f1_ReLU, f2_ReLU])

      f1_tanh = hyp_tan(z(inputs[i], weights_1[0], bias[0]))
      f2_tanh = hyp_tan(z(inputs[i], weights_2[1], bias[1]))

      results['tanh'].append([f1_tanh, f2_tanh])

      f1_z = z(inputs[i], weights_1[0], bias[0])
      f2_z = z(inputs[i], weights_2[1], bias[1])

      results['z'].append([f1_z, f2_z])

    for f in results.keys():
      fig = plt.figure()
      plt.title(f)
      for r in range(len(results[f])):
        color = 'blue' if preds[r] == 1 else 'red'
        plt.scatter(results[f][r][0], results[f][r][1], marker='.', color=color)
      plt.xlabel('f1')
      plt.ylabel('f2')

  plt.show()

# Discrete Convolution
if True:
  def convolve(f, g):
    N = len(f)
    M = len(g)
    h = [0] * M
    for n in range(M):
      for k in range(N):
        if n-k >= 0 and n-k < N:
          h[n] += f[k] * g[n-k]
    return h

  f = [1, 3, -1, 1, -3]
  g = [0, 1, 0, -1, 0]
  h = convolve(f, g)
  print(h)

embed()