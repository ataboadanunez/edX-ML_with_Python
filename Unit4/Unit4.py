import numpy as np
from numpy.random import uniform
import random
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed

def cost(x, z):
	"""
		Computes the clustering cost given a feature vector (x) and reprensentative (z) based on euclidean distance

		Args: 
			x (numpy array): feature vector
			z (numpy array): representative
	"""

	return np.linalg.norm(x - z)

def cost_l1(x1, x2):

	return np.linalg.norm(x1 -x2, ord=1)

# Exercise Calculating Cost
# consider that x1, x2, x3 correspond to Cluster1 with representative z1
# and the feature vectors x4, x5 correspond to Cluster2 with representative z2

# Given the values of the points below, calculate:
# a) cost of C1
# b) cost of C2
# c) cost of (C1, C2)

x1 = np.array([-1, 2])
x2 = np.array([-2, 1])
x3 = np.array([-1, 0])
x4 = np.array([2, 1])
x5 = np.array([3, 2])

z1 = np.array([-1, 1])
z2 = np.array([2, 2])

C1 = [x1, x2, x3]
C2 = [x4, x5]

cost_C1 = 0
cost_C2 = 0
for x in C1:
	cost_C1 += cost(x, z1)

for x in C2:
	cost_C2 += cost(x, z2)

cost_C1C2 = cost_C1 + cost_C2

print("a) Cost(C1) = %.4f" %cost_C1)
print("b) Cost(C2) = %.4f" %cost_C2)
print("c) Cost(C1, C2) = %.4f" %cost_C1C2)


# Gaussian Distribution

def gaussian_likelihood(x, mu, sigma):
	d = x.shape[0]
	dist = np.linalg.norm(x - mu)

	return (1 / (2*np.pi*sigma**2)**(d / 2)) * np.exp(-dist / (2 * sigma**2))


x_arr = np.array([1/np.sqrt(np.pi), 2])
sigma = np.sqrt(1 / (2*np.pi))
mu_arr = np.array([0, 2])

gaussian_likelihood = gaussian_likelihood(x_arr, mu_arr, sigma)
log_gaussian_likelihood = np.log(gaussian_likelihood)
print("Gaussian Likelihood = %.4f" %gaussian_likelihood)
print("Log(Gaussian_Likelihood) = %.4f" %log_gaussian_likelihood)

# Compute mean of the following clusters
x1_1 = np.array([-1.2, -0.8])
x2_1 = np.array([-1, -1.2])
x3_1 = np.array([-0.8, -1])

x1_2 = np.array([1.2, 0.8])
x2_2 = np.array([1., 1.2])
x3_2 = np.array([0.8, 1])

print("mu_1 = ", np.mean([x1_1, x2_1, x3_1], axis=0))
print("mu_2 = ", np.mean([x1_2, x2_2, x3_2], axis=0))


# Gaussian Mixture Model: An Example Update - E-Step
def normal_dist(x , mean , sd):
		sd = np.sqrt(sd)
		prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
		return prob_density

p1 = 0.5
p2 = 0.5
mu1 = -3
mu2 = 2
s1 = 4
s2 = 4

x1 = 0.2
x2 = -0.9
x3 = -1
x4 = 1.2
x5 = 1.8

N11 = normal_dist(x1, mu1, s1)
N21 = normal_dist(x2, mu1, s1)
N31 = normal_dist(x3, mu1, s1)
N41 = normal_dist(x4, mu1, s1)
N51 = normal_dist(x5, mu1, s1)

N12 = normal_dist(x1, mu2, s2)
N22 = normal_dist(x2, mu2, s2)
N32 = normal_dist(x3, mu2, s2)
N42 = normal_dist(x4, mu2, s2)
N52 = normal_dist(x5, mu2, s2)

D1 = p1*N11 + p2*N12
D2 = p1*N21 + p2*N22
D3 = p1*N31 + p2*N32
D4 = p1*N41 + p2*N42
D5 = p1*N51 + p2*N52

p11 = (p1 * N11) / D1
p12 = (p1 * N21) / D2
p13 = (p1 * N31) / D3
p14 = (p1 * N41) / D4
p15 = (p1 * N51) / D5

print("p(1|1) = %.5f" %p11)
print("p(1|2) = %.5f" %p12)
print("p(1|3) = %.5f" %p13)
print("p(1|4) = %.5f" %p14)
print("p(1|5) = %.5f" %p15)

p1_up = (p11 + p12 + p13 + p14 + p15) / 5
mu1_up = (p11*x1 + p12*x2 + p13*x3 + p14*x4 + p15*x5) / (p11 + p12 + p13 + p14 + p15)
s1_up = (p11 * np.linalg.norm(x1 - mu1_up) + p12 * np.linalg.norm(x2 - mu1_up) + p13 * np.linalg.norm(x3 - mu1_up) + p14 * np.linalg.norm(x4 - mu1_up) + p15 * np.linalg.norm(x5 - mu1_up)) / ((p11 + p12 + p13 + p14 + p15))
print("----------------------------")
print("Updated parameters: ")
print("p1' = %.5f" %p1_up)
print("mu1' = %.5f" %mu1_up)
print("sig1' = %.5f" %s1_up)


# HOMEWORK 4
# 1. K-means and K-medoids
"""
	Assume we have a 2D dataset consisting of (0, -6), (4, 4), (0, 0), (-5, 2). We wish to do k-means and k-medoids clustering with k=2. We initialize the cluster centers with (-5, 2), (0, -6).
	For this small dataset, in choosing between two equally valid exemplars for a cluster in k-medoids, choose them with priority in the order given above (i.e. all other things being equal, you would choose (0, -6) as a center over (-5, 2)).

	For the following scenarios, give the clusters and cluster centers after the algorithm converges. Enter the coordinate of each cluster center as a square-bracketed list (e.g. [0, 0]); enter each cluster's members in a similar format, separated by semicolons (e.g. [1, 2]; [3, 4]). 
"""

# First draw data-points
X_train = np.array([[0., -6.], [4., 4.], [0., 0.], [-5., 2.]])
cen_1 = np.array([-5., 2.])
cen_2 = np.array([0., -6.])
fig = plt.figure()
sns.scatterplot(x=[X[0] for X in X_train],
								y=[X[1] for X in X_train],
								palette="deep",
								legend=None)

plt.xlabel(r'x')
plt.ylabel(r'y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)


# calculate cluster centers using sklearn KMeans and KMedoids
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_train)
print(kmeans.labels_)
kmeans.predict([cen_1, cen_2])
print("Cluster centers using KMeans: ")
cluster_cen_1, cluster_cen_2 = kmeans.cluster_centers_
print("Cluster center 1 = ", cluster_cen_1)
print("Cluster center 2 = ", cluster_cen_2)

print("---------------------------------------")
print("Cluster centers using KMedoids: ")

# 3. EM Algorithm
x0 = -1
x1 = 0
x2 = 4
x3 = 5
x4 = 6

pi1 = 0.5
pi2 = 0.5
mu1 = 6
mu2 = 7 
sigma1 = 1
sigma2 = 4

X = [x0, x1, x2, x3, x4]
p = 0

def p_k(x, k):
	if k == 1:
		pik = pi1
		Nk = normal_dist(x, mu1, sigma1)
	elif k == 2:
		pik = pi2
		Nk = normal_dist(x, mu2, sigma2)

	D = pi1 * normal_dist(x, mu1, sigma1) + pi2 * normal_dist(x, mu2, sigma2)

	return (pik * Nk) / D


for x in X:
	p += pi1*normal_dist(x, mu1, sigma1) + pi2*normal_dist(x, mu2, sigma2)

print("Log Likelihood = %i" %round(np.log(p)) )
print("p(2|x0) > p(1|x0) = ", (p_k(x0, 2) > p_k(x0, 1)))
print("p(2|x1) > p(1|x1) = ", (p_k(x1, 2) > p_k(x1, 1)))
print("p(2|x2) > p(1|x2) = ", (p_k(x2, 2) > p_k(x2, 1)))
print("p(2|x3) > p(1|x3) = ", (p_k(x3, 2) > p_k(x3, 1)))
print("p(2|x4) > p(1|x4) = ", (p_k(x4, 2) > p_k(x4, 1)))

embed()
if False:

	# Alternative way
	def euclidean(point, data):
			"""
			Return euclidean distances between a point & a dataset
			"""
			return np.sqrt(np.sum((point - data)**2, axis=1))


	class KMeansGH:

			def __init__(self, n_clusters=8, max_iter=300):
					self.n_clusters = n_clusters
					self.max_iter = max_iter

			def fit(self, X_train):

					# Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
					# then the rest are initialized w/ probabilities proportional to their distances to the first
					# Pick a random point from train data for first centroid
					self.centroids = [random.choice(X_train)]

					for _ in range(self.n_clusters-1):
							# Calculate distances from points to the centroids
							dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
							# Normalize the distances
							dists /= np.sum(dists)
							# Choose remaining points based on their distances
							new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)[0]  # Indexed @ zero to get val, not array of val
							self.centroids += [X_train[new_centroid_idx]]

					# This method of randomly selecting centroid starts is less effective
					# min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
					# self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

					# Iterate, adjusting centroids until converged or until passed max_iter
					iteration = 0
					prev_centroids = None
					while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
							# Sort each datapoint, assigning to nearest centroid
							sorted_points = [[] for _ in range(self.n_clusters)]
							for x in X_train:
									dists = euclidean(x, self.centroids)
									centroid_idx = np.argmin(dists)
									sorted_points[centroid_idx].append(x)

							# Push current centroids to previous, reassign centroids as mean of the points belonging to them
							prev_centroids = self.centroids
							self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
							for i, centroid in enumerate(self.centroids):
									if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
											self.centroids[i] = prev_centroids[i]
							iteration += 1

			def evaluate(self, X):
					centroids = []
					centroid_idxs = []
					for x in X:
							dists = euclidean(x, self.centroids)
							centroid_idx = np.argmin(dists)
							centroids.append(self.centroids[centroid_idx])
							centroid_idxs.append(centroid_idx)

					return centroids, centroid_idxs


	kmeans_gh = KMeansGH(n_clusters=2)
	kmeans_gh.fit(X_train)
	class_centers, classification = kmeans_gh.evaluate(X_train)
	print(class_centers)
