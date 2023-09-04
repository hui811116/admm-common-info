import numpy as np
import os
import sys


def rowOneMat(px1x2):
	ns = px1x2.shape
	nx1 = ns[0]
	nx2 = ns[1]
	one_vec = np.ones((nx1 * nx2 ,1 ))
	return one_vec @ one_vec.T

def diagMat(px1x2):
	ns = px1x2.shape
	nx1 = ns[0]
	nx2 = ns[1]
	Amat = np.eye(nx1*nx2)
	for xx2 in range(nx2):
		for xx1 in range(nx1):
			Amat[xx2*nx1 + xx1,xx2*nx1 + xx1] = px1x2[xx1,xx2]
	return Amat

def partialMat(px1,px2,diagonal):
	# sum out the last axis always?
	ns1 = px1.shape[0]
	ns2 = px2.shape[0]
	Amat = np.zeros((ns1*ns2,ns1*ns2))
	if diagonal:
		# px1cz
		for xx2_a in range(ns2):
			for xx1 in range(ns1):
				idx_row = xx1 + xx2_a * ns1
				for xx2_b in range(ns2):
					idx_col = xx1 + xx2_b*ns1
					Amat[idx_row,idx_col] = px1[xx1]
	else:
		# px2cz
		# off-diagonal version
		for xx1_a in range(ns1):
			for xx2 in range(ns2):
				idx_row = xx1_a + xx2 * ns1
				for xx1_b in range(ns1):
					idx_col = xx1_b + xx2* ns1
					Amat[idx_row,idx_col] = px2[xx2] 
	return Amat


def computeHessian_2views(pzcx1x2,px1x2,**kwargs):
	# only for two views
	px1 = np.sum(px1x2,axis=1)
	px2 = np.sum(px1x2,axis=0)

	pzx1x2 = pzcx1x2 * px1x2[None,:,:]
	pz = np.sum(pzx1x2,axis=(1,2))

	pzx1 = np.sum(pzx1x2,axis=2)
	pzx2 = np.sum(pzx1x2,axis=1)

	px1x2cz = np.transpose(pzx1x2/pz[:,None,None],axes=(1,2,0))

	px1cz = (pzx1/ pz[:,None]).T
	px2cz = (pzx2/ pz[:,None]).T
	# see derivations
	ns = pzcx1x2.shape
	nz = ns[0]
	nx1 = ns[1]
	nx2 = ns[2]
	com_eig = np.zeros((nz,nx1*nx2))
	for zz in range(nz):
		#Amat = diagMat(1/px1x2cz[:,:,zz]) # the reciprocal
		#row_one_mat = rowOneMat(px1x2cz[:,:,zz])
		X1mat = partialMat(1/px1cz[:,zz],1/px2cz[:,zz],diagonal=True)
		X2mat = partialMat(1/px1cz[:,zz],1/px2cz[:,zz],diagonal=False)

		Wmat = np.diag(np.sqrt(np.squeeze(np.reshape(px1x2cz[:,:,zz],(nx1*nx2,)).T)))
		Emat = 0.5 * Wmat @ (X1mat + X2mat) @ Wmat
		eigval, eigvec = np.linalg.eigh(Emat)
		#print("z{:}".format(zz))
		#print(eigval)
		srt_eigval = np.sort(eigval)
		com_eig[zz,:] = srt_eigval
	return com_eig