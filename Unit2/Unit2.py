import numpy as np
from IPython import embed

def Loss1(z):
	if z>= 1:
		loss = 0.
	else:
		loss = 1. - z
	return loss

def Loss2(z):
	loss = 0.5 * z**2
	return loss

def Risk(loss_function, feature_matrix, labels, theta, theta_0=0):
	
	risk = []
	for (feature_vector, label) in zip(feature_matrix, labels):
		z = label - (np.dot(theta, feature_vector) + theta_0)
		risk.append(loss_function(z))

	risk = np.array(risk)

	return np.mean(risk)


# compute empirical risk
x1 = np.array([1, 0, 1])
x2 = np.array([1, 1, 1])
x3 = np.array([1, 1, -1])
x4 = np.array([-1, 1, 1])

y1 = 2
y2 = 2.7
y3 = -0.7
y4 = 2

theta = np.array([0, 1, 2])

embed()