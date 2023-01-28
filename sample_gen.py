import numpy as np
import dataset as dt
import sys
import os
import pickle

rng = np.random.default_rng(seed=1234)

ny = 8
nb = 2
corr =0.2    # 0.2, seed 1234, 0.0, seed 4321
datadict = dt.synExpandToy(ny,nb,corr)
if not "py" in datadict.keys():
	sys.exit("Dataset {:} has no hidden labels".format(dataname))

py = datadict["py"]
px1cy = datadict["pycx_list"][0]
px2cy = datadict["pycx_list"][1]

px1 = np.sum(px1cy* py[None,:],1)
px2 = np.sum(px2cy * py[None,:],1)
px12 = np.zeros((px1cy.shape[0],px2cy.shape[0]))
for ix1 in range(len(px1)):
	for ix2 in range(len(px2)):
		tmp_sum = 0
		for iy in range(len(py)):
			tmp_sum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
		px12[ix1,ix2]= tmp_sum

n_samp = 100000 # for training
n_train = 10000 # for testing 
def inverseTransformSampling(num_samp,py,px1cy,px2cy):
	label_samp = rng.random((num_samp,))
	ymap = np.cumsum(py)
	ylabel = -1 * np.ones((num_samp,))
	for idx,prob in enumerate(label_samp):
		for yi in range(len(ymap)):
			if prob < ymap[yi]:
				ylabel[idx] = yi
				break
	if np.any(ylabel<0):
		sys.exit("ERROR: some samples have no label")
	ylabel = ylabel.astype("int")
	# get observations
	x_prob = rng.random((num_samp,2)) # 2 views
	x_sample = -1 * np.ones((num_samp,2))
	for idx in range(num_samp):
		ytmp = ylabel[idx]
		prob1 = x_prob[idx,0]
		prob2 = x_prob[idx,1]
		x1_map = np.cumsum(px1cy[:,ytmp])
		for xi in range(len(px1)):
			if prob1 < x1_map[xi]:
				x_sample[idx,0] = xi
				break
		x2_map = np.cumsum(px2cy[:,ytmp])
		for xi in range(len(px2)):
			if prob2 < x2_map[xi]:
				x_sample[idx,1] = xi
				break
	if np.any(x_sample<0):
		sys.exit("ERROR: some observations have no realization")
	x_sample = x_sample.astype("int")
	return {"ylabel":ylabel,"xsample":x_sample,"ny":len(py),"nx1":px1cy.shape[0],"nx2":px2cy.shape[0]}
# inverse transform sampling
'''
label_samp = rng.random((n_samp,))
y_map = np.cumsum(py)
y_label = -1 * np.ones((n_samp,))
for idx, prob in enumerate(label_samp):
	for yi in range(len(y_map)):
		if prob < y_map[yi]:
			y_label[idx] = yi
			break
if np.any(y_label<0):
	sys.exit("ERROR: some samples have no label")

y_label = y_label.astype("int")

x_prob = rng.random((n_samp,2)) # 2 views
x_sample = -1 * np.ones((n_samp,2))
for idx in range(n_samp):
	ytmp = y_label[idx]
	prob1 = x_prob[idx,0]
	prob2 = x_prob[idx,1]
	# for x1cy
	x1_map = np.cumsum(px1cy[:,ytmp])
	for xi in range(len(px1)):
		if prob1 < x1_map[xi]:
			x_sample[idx,0] = xi
			break
	x2_map = np.cumsum(px2cy[:,ytmp])
	for xi in range(len(px2)):
		if prob2 < x2_map[xi]:
			x_sample[idx,1] = xi
			break

if np.any(x_sample<0):
	sys.exit("ERROR: some observations have no realization")
x_sample = x_sample.astype("int")
'''
train_dict = inverseTransformSampling(n_train,py,px1cy,px2cy)
test_dict = inverseTransformSampling(n_samp,py,px1cy,px2cy)
with open("opt_y{:}b{:}_cr{:.4e}_dataset.pkl".format(ny,nb,corr),'wb') as fid:
	pickle.dump({"train_dict":train_dict,"test_dict":test_dict},fid)

# all passed
'''
with open("test_opt_y{:}b{:}_cr{:.4e}_label.npy".format(ny,nb,corr),'wb') as fid:
	np.save(fid,y_label)
with open("test_opt_y{:}b{:}_cr{:.4e}_obs.npy".format(ny,nb,corr),"wb") as fid:
	np.save(fid,x_sample)
'''