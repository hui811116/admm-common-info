import numpy as np
import itertools
from numpy.random import default_rng

import os
import sys
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import itertools

from sklearn.cluster import KMeans
from sklearn import datasets
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix, adjusted_rand_score, v_measure_score, accuracy_score

from munkres import Munkres


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

#### from WVAE-Torch

def compute_metrics(y_true,y_pred):
	out_nmi = nmi(y_true,y_pred)
	out_ari = ari(y_true,y_pred)
	out_vms = vmeasure(y_true,y_pred)
	out_pur = purity(y_true,y_pred)
	out_acc, out_acc_cnt, out_total_cnt = clustering_accuracy(y_true,y_pred)
	return {
		"nmi":out_nmi,
		"ari":out_ari,
		"vmeasure":out_vms,
		"purity":out_pur,
		"acc":out_acc,
		"acc_cnt":int(out_acc_cnt),
		"total_cnt":out_total_cnt,
	}

def nmi(y_true, y_pred):
	return normalized_mutual_info_score(y_true,y_pred)
def ari(y_true, y_pred):
	return adjusted_rand_score(y_true,y_pred)
def vmeasure(y_true, y_pred):
	return v_measure_score(y_true, y_pred)

def purity(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """

    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1]
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)
    '''
    cm = confusion_matrix(y_true, y_pred)
    row_max = cm.max(axis=1).sum()
    total = cm.sum()
    pur = row_max / total
    return pur
	'''

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    acc_cnt = sum([w[i, j] for i, j in ind]) * 1.0
    total_cnt = y_pred.size
    return acc_cnt / y_pred.size, acc_cnt, total_cnt

def compare_global_solution(pycx1x2,pzcx1x2):
	assert pycx1x2.shape == pzcx1x2.shape
	ny = pycx1x2.shape[0]
	w = np.zeros((ny,ny),dtype=np.int64)
	for iy in range(ny):
		for iz in range(ny):
			# compute the cost
			w[iy,iz] = np.sum(np.fabs(pycx1x2[iy,:,:]-pzcx1x2[iz,:,:]))
	row_ind,col_ind = linear_sum_assignment(w)
	
	matched_pzcx1x2 = np.zeros(pzcx1x2.shape)
	for nd in range(ny):
		rz = row_ind[nd]
		cy = col_ind[nd]
		matched_pzcx1x2[cy,:,:] = pzcx1x2[rz,:,:]
	dtv_corrected = np.sum(np.fabs(matched_pzcx1x2-pycx1x2),axis=0)
	return matched_pzcx1x2, dtv_corrected

def labelAssignFromMap(pred_2dmap):
	# must be square, must be 2d map
	assert len(pred_2dmap.shape)==2
	assert pred_2dmap.shape[0] == pred_2dmap.shape[1]
	# the map should be in y(row), z(col) format
	assign_dict = {}
	while True:
		unassigned = []
		as_zs = []
		for idx in range(pred_2dmap.shape[0]):
			if not idx in assign_dict.keys():
				unassigned.append(idx)
			else:
				as_zs.append(assign_dict[idx]['zsymbol'])
		if len(unassigned) == 0:
			break
		# every step assign one, with the highest likelihood
		high_like = -1
		high_yarg = -1
		cand_zarg = -1
		for idy in unassigned:
			for zzid in range(pred_2dmap.shape[0]):
				if not zzid in as_zs:
					if pred_2dmap[idy,zzid] > high_like:
						high_like = pred_2dmap[idy,zzid]
						high_yarg = idy
						cand_zarg = zzid
		if high_like >= 0.0:
			assign_dict[high_yarg] ={"zsymbol":cand_zarg,"ylabel":high_yarg,'likelihood':high_like}
		else:
			sys.exit("ERROR: no assignment found")
	# return a z_to_y and y_to_z map
	z2y_map = {}
	y2z_map = {}
	for k,v in assign_dict.items():
		z2y_map[v['zsymbol']] = v['ylabel']
		y2z_map[v['ylabel']]  = v['zsymbol']
	return {"z2y_map":z2y_map,"y2z_map":y2z_map}

def postCalcAccuracy(z_train,y_train,z_test,y_test):
	#z_train, y_train = concatPred(model,train_dataset)
	#z_test, y_test = concatPred(model,test_dataset)
	# NOTE:
	# z_train: concatenated z_predictions,
	# y_train: concatenated y_labels
	# so and so forth...
	#
	classifier = LogisticRegression(solver="saga",multi_class="multinomial")
	classifier.fit(z_train,y_train)
	train_acc = classifier.score(z_train,y_train)
	test_acc = classifier.score(z_test,y_test)
	return train_acc, test_acc

## revised from the siolag161 implementation

def make_cost_matrix(c1,c2):
	"""
	"""
	uc1 = np.unique(c1)
	uc2 = np.unique(c2)
	l1 = uc1.size
	l2 = uc2.size
	assert(l1==l2 and np.all(uc1==uc2))

	m = np.ones([l1,l2])
	for i in range(l1):
		it_i = np.nonzero(c1 == uc1[i])[0]
		for j in range(l2):
			it_j = np.nonzero(c2==uc2[j])[0]
			m_ij = np.intersect1d(it_j,it_i)
			m[i,j] = -m_ij.size
	return m

def translate_clustering(clt, mapper):
	return np.array([mapper[i] for i in clt])

def accuracy(cm):
	"""computes accuracy from confusion matrix"""
	return np.trace(cm,dtype=float) / np.sum(cm)


def evaluateWithMap(z_pred,y_label,mapper):
	z_est = [mapper[item] for item in z_pred]
	acc_cnt = np.sum(z_est == y_label)
	return acc_cnt/ len(y_label), acc_cnt, len(y_label)

def getLabelMap(z_pred,y_label):
	num_labels = len(np.unique(y_label))
	#cm = confusion_matrix(y_label,z_args, labels=range(num_labels))
	cost_matrix = make_cost_matrix(z_pred,y_label)
	m = Munkres()
	indexes = m.compute(cost_matrix)
	mapper = {old: new for (old,new) in indexes}
	return mapper

