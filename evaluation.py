import numpy as np
import sys
import os
import itertools

# here we only check the case where |Z|=|Y|
# Question: How to map |Z| -> |Y| without knowing Y...
def clusteringTest2View(y_test,x_test,enc_vv):
	rng = np.random.default_rng()

	# using Bayes' decoder for clustering
	x_prob = rng.random(y_test.shape)
	z_out = -1 * np.ones(y_test.shape)
	for idx in range(len(y_test)):
		tmp_prob = x_prob[idx]
		tmp_y = y_test[idx]
		tmp_x1 = x_test[idx,0]
		tmp_x2 = x_test[idx,1]

		inv_map = np.cumsum(enc_vv[:,tmp_x1,tmp_x2])
		for iy in range(len(inv_map)):
			if tmp_prob < inv_map[iy]:
				z_out[idx] = iy
				break
	z_out = z_out.astype("int")
	if np.any(z_out<0):
		sys.exit("ERROR: some testing data have no cluster")

	# account for the permutation of all possible labeling
	perm_list = list(itertools.permutations(np.arange(0,enc_vv.shape[0])))
	acc_list = np.zeros((len(perm_list),))
	for pidx,item in enumerate(perm_list):
		new_lab = np.zeros(y_test.shape)
		for nidx, inner in enumerate(z_out):
			new_lab[nidx] = item[inner]
		acc_list[pidx] = np.sum(new_lab == y_test)
	'''
	print(enc_vv)
	print(y_test[:20])
	print(z_out[:20])
	print(acc_list)
	sys.exit()
	'''
	optmap = perm_list[np.argmax(acc_list)]
	out_res = np.zeros(y_test.shape)
	for idx in range(len(y_test)):
		out_res[idx] = optmap[z_out[idx]]
	acc_rate = np.amax(acc_list)/len(y_test)
	return out_res, acc_rate

