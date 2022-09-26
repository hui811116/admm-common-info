import numpy as np
import dataset as dt
import sys
import os

rng = np.random.default_rng()

#datadict = dt.getDataset("condindp2v")
dataname = "toy2v"
datadict = dt.getDataset(dataname)
#print(datadict)
if not "py" in datadict.keys():
	sys.exit("Dataset {:} has no hidden labels".format(dataname))

'''
# CondIndp
py = np.array([0.25,0.40,0.35])
px1cy = np.array([
	[0.85, 0.04 , 0.06],
	[0.07, 0.40 , 0.60],
	[0.08, 0.56 , 0.34]])
px2cy = np.array([
	[0.30, 0.08, 0.35],
	[0.40, 0.80, 0.15],
	[0.30, 0.12, 0.50]])
'''
# Overlap
'''
py = np.array([0.25,0.40,0.35])
px1cy = np.array([
	[0.85, 0.04 , 0.06],
	[0.07, 0.40 , 0.80],
	[0.08, 0.56 , 0.14]])
px2cy = np.array([
	[0.30, 0.08, 0.05],
	[0.40, 0.80, 0.05],
	[0.30, 0.12, 0.90]])
'''
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

n_samp = 5000
# inverse transform sampling
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


# all passed
with open("test_{:}_label.npy".format(dataname),'wb') as fid:
	np.save(fid,y_label)
with open("test_{:}_obs.npy".format(dataname),"wb") as fid:
	np.save(fid,x_sample)