# code for project0
import numpy as np
import pdb
from IPython import embed

def randomization(n):
	"""
	Arg:
		n - an integer
	Returns:
		A - a randomly-generated nx1 Numpy array
	"""
	A = np.random.rand(n, 1)

	return A

	raise NotImplementedError

def operations(h, w):
	"""
	Takes two inputs, h and w, and makes two Numpy arrays A and B of size
		h x w, and returns A, B, and s, the sum of A and B.

	Arg:
		h - an integer describing the height of A and B
		w - an integer describing the width of A and B
	Returns (in this order):
		A - a randomly-generated h x w Numpy array.
		B - a randomly-generated h x w Numpy array.
		s - the sum of A and B.

	"""

	A = np.random.rand(h,w)
	B = np.random.rand(h,w)
	s = A + B

	return A, B, s


def norm(A, B):
	"""
	Takes two Numpy column arrays, A and B, and returns the L2 norm of their
	sum.

	Arg:
		A - a Numpy array
		B - a Numpy array
	Returns:
		s - the L2 norm of A+B.
	"""
	S = A + B
	s = np.linalg.norm(S)

	return s

	raise NotImplementedError


def neural_network(inputs, weights):
	"""
	 Takes an input vector and runs it through a 1-layer neural network
	 with a given weight matrix and returns the output.

	 Arg:
		 inputs - 2 x 1 NumPy array
		 weights - 2 x 1 NumPy array
	 Returns (in this order):
		 out - a 1 x 1 NumPy array, representing the output of the neural network
	"""

	out = np.tanh(np.dot(weights.T, inputs))

	return out

	raise NotImplementedError


def scalar_function(x, y):
	"""
		Returns f(x,y) where
		f(x, y) = {x * y if x<= y; x/y else}

	"""

	def f(x, y):
		if (x <= y):
			return x*y
		else:
			return x/y

	return f(x,y)

	raise NotImplementedError

def vector_function(x, y):

	def f(x, y):
		if (x <= y):
			return x*y
		else:
			return x/y

	vec_f = np.vectorize(f)

	return vec_f(x,y)

def get_sum_metrics(predictions, metrics=[]):
		
	list_metrics_values = [metric(predictions) for metric in metrics]
	sum_metrics = sum(list_metrics_values)
	sum_metrics += 3*predictions+3

	return sum_metrics


if False:
	print("Calculating output of neural_network: ")
	inputs = np.random.rand(2,1)
	weights = np.random.rand(2,1)

	out = neural_network(inputs, weights)
	print("Input vector = ", inputs)
	print("Weights = ", weights)
	print("Output = ", out)


if False:
	print("Scalar function")
	x = np.random.rand()
	y = np.random.rand()
	out = scalar_function(x, y)

	print("x = ", x)
	print("y = ", y)
	print("scalar_function = ", out)

if False:
	print("Vector function")
	x = np.random.rand(2,1)
	y = np.random.rand(2,1)
	out = vector_function(x, y)

	print("x = ", x)
	print("y = ", y)
	print("scalar_function = ", out)

if True:
	s = get_sum_metrics(1)
embed()