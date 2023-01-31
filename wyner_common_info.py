import numpy as np
import sys
import os
import matplotlib.pyplot as plt

rng = np.random.default_rng()

nz = 2
nx1 = 3
nx2 = 4

pz = rng.random((nz,))
pz /= np.sum(pz)
print(pz)
px1cz = rng.random((nx1,nz))
px1cz = px1cz / np.sum(px1cz,axis=0,keepdims=True)
print(np.sum(px1cz,axis=0))
px2cz = rng.random((nx2,nz))
px2cz = px2cz / np.sum(px2cz,axis=0,keepdims=True)
print(np.sum(px2cz,axis=0))

# raw combination 
expand_px1cz = np.repeat(np.expand_dims(px1cz,axis=1),repeats=nx2,axis=1)
expand_px2cz = np.repeat(np.expand_dims(px2cz,axis=0),repeats=nx1,axis=0)
'''
est_px1x2 = np.zeros((nx1,nx2))
for ii1 in range(nx1):
	for ii2 in range(nx2):
		ele_sum =0
		for iz in range(nz):
			ele_sum += px1cz[ii1,iz] * px2cz[ii2,iz] * pz[iz]
		est_px1x2[ii1,ii2] = ele_sum
'''
est_px1x2 = np.sum(expand_px1cz*expand_px2cz*pz[...,:],axis=-1)
print(est_px1x2)
print(np.sum(est_px1x2))
# this maps to almost 1.0, with some round-off error
# condition encoder
joint_est = expand_px1cz * expand_px2cz*pz[...,:]
joint_est = np.transpose(joint_est,axes=[2,0,1]) # z x1 x2
est_pzcx1x2 = joint_est / np.sum(joint_est,axis=0,keepdims=True)
print(est_pzcx1x2)