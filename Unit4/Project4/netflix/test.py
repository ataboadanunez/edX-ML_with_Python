import numpy as np
import em
import common
from IPython import embed
X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 6
n, d = X.shape
seed = 0

from naive_em import estep, mstep, run
init_mix, init_post = common.init(X, K, seed)

# test E-step function
post, likelihoods = estep(X, init_mix)
print(post, likelihoods)
# test M-step function
new_mix = mstep(X, post)
print(new_mix)
# test run function
res = run(X, init_mix, init_post)
embed()