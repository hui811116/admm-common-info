import numpy as np
import sys
import os
import itertools
from sklearn.linear_model import LogisticRegression
from numpy.random import default_rng

# here we only check the case where |Z|=|Y|
# Question: How to map |Z| -> |Y| without knowing Y...
'''
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

	# NOTE: instead of exhaustive search over all combinations
	#       use linear regression to find the best mapping (linear classifier)
	#       then report the accuracy from the learned mapping
	#       this follows the common practice in unsupervised multiview learning
	#       cite: learning robust representation through multiview information bottleneck
	#             contrastivce multiview coding
	#train_acc = 
	# using soft probability as prediction
	#test_acc = postCalculateAccuracy()
	# using sampled z and make it one hot (hard prediction)
	#test_hard_acc = 
	test_acc =0
	test_hard_acc =0
	
	return test_acc, test_hard_acc 
'''
def labelSampling(y_label,x_sample,z_enc):
	rng = default_rng()
	nsamp = y_label.shape[0]
	z_prob = rng.random(nsamp)
	z_out = -1 * np.ones(nsamp)
	for idx in range(nsamp):
		tmp_prob = z_prob[idx]
		tmp_y = y_label[idx]
		tmp_x1 = x_sample[idx,0]
		tmp_x2 = x_sample[idx,1]
		inv_map = np.cumsum(z_enc[:,tmp_x1,tmp_x2])
		for iy in range(len(inv_map)):
			if tmp_prob < inv_map[iy]:
				z_out[idx] = iy
				break
	z_out = z_out.astype("int32")
	if np.any(z_out)<0:
		sys.exit("ERROR: some testing data have no cluster")
	return z_out

def oneHot(zlabel,num_dim):
	nsamp = zlabel.shape[0]
	output = np.zeros((nsamp,num_dim))
	for idx, item in enumerate(zlabel):
		output[idx,item] = 1.0
	return output

def evalSamples2V(y_train,x_train,y_test,x_test,z_enc):
	rng = default_rng()
	ylist = []
	for ele in y_train:
		if not ele in ylist:
			ylist.append(ele)
	ny = len(ylist)
	z_train = labelSampling(y_train,x_train,z_enc)
	z_test = labelSampling(y_test,x_test,z_enc)
	z_train_one = oneHot(z_train,ny)
	z_test_one = oneHot(z_test,ny)
	# use the trained classifier
	# fitted afterward
	classifier = LogisticRegression(solver="saga",multi_class="multinomial").fit(z_train_one,y_train)
	# use the fitted classifier to score
	train_acc = classifier.score(z_train_one,y_train)
	test_acc = classifier.score(z_test_one,y_test)
	return train_acc, test_acc
