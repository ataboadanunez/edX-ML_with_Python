import numpy as np
import kmeans
import common
import naive_em
import em

from collections import defaultdict

from IPython import embed

X = np.loadtxt("toy_data.txt")


"""
	2. K-means
	For this part of the project you will compare clustering obtained via K-means to the (soft) clustering induced by EM. In order to do so, our K-means algorithm will differ a bit from the one you learned. Here, the means are estimated exactly as before but the algorithm returns additional information. More specifically, we use the resulting clusters of points to estimate a Gaussian model for each cluster. Thus, our K-means algorithm actually returns a mixture model where the means of the component Gaussians are the K centroids computed by the K-means algorithm. This is to make it such that we can now directly plot and compare solutions returned by the two algorithms as if they were both estimating mixtures. 

	Read a 2D toy dataset using X = np.loadtxt('toy_data.txt'). Your task is to run the K-means algorithm on this data using the implementation we have provided in kmeans.py Initialize K-means using common.init(X, K, seed), where is the number of clusters and seed is the random seed used to randomly initialize the parameters. 

"""
if True:
	print("Exercise K-Means")
	result = defaultdict(list)
	BIC_dic = defaultdict(list)
	Ks = [1, 2, 3, 4]
	seeds = [0, 1, 2, 3, 4]

	for K in Ks:
		k_best_mix, k_best_post, k_best_cost = None, None, np.inf
		em_best_mix, em_best_post, em_best_ll = None, None, -np.inf

		for seed in seeds:

			init_mixture, init_post = common.init(X, K, seed)
			k_mixture, k_post, k_cost = kmeans.run(X, init_mixture, init_post)
			em_mixture, em_post, em_ll = naive_em.run(X, init_mixture, init_post)

			if k_cost < k_best_cost:
				k_best_cost = k_cost
				k_best_mix = k_mixture
				k_best_post = k_post
			
			if em_ll > em_best_ll:
				em_best_ll = em_ll
				em_best_mix = em_mixture
				em_best_post = em_post

			result['cost_K_%i' %K].append(k_cost)
			result['ll_K_%i' %K].append(em_ll)

		BIC_dic['K_%i' %K].append(common.bic(X, em_best_mix, em_best_ll))
		#common.plot(X, k_best_mix, k_best_post, "K-means K = %i" %K)
		#common.plot(X, em_best_mix, em_best_post, "EM K = %i" %K)

	# print the lowest cost for each K
	for item in result.keys():
		print("Minimum %s = %.4f" %(item, min(result[item])))


embed()