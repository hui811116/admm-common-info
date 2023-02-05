import numpy as np
import sys
import os
from scipy.special import softmax

def calcMI(pxy):
	return np.sum(pxy * np.log(pxy/np.sum(pxy,0,keepdims=True)/np.sum(pxy,1,keepdims=True)))

def calcMIcond(pxyz):
	pz = np.sum(pxyz,axis=(0,1),keepdims=True)
	pxycz = pxyz/pz
	pxcz = np.sum(pxycz,axis=1,keepdims=True)
	pycz = np.sum(pxycz,axis=0,keepdims=True)
	return max(np.sum(pxyz * (np.log(pxycz) - np.log(pycz) - np.log(pxcz)) ),0)

def calcEnt(pz):
	return -np.sum(pz * np.log(pz))

def calcCondEnt(pzx):
	return -np.sum(pzx * np.log(pzx/np.sum(pzx,axis=0,keepdims=True)))

def aggregateResults(xdata,ydata,criterion,precision):
	precision_tex = "{{:.{:}f}}".format(precision)
	out_dict = {}
	for idx in range(len(xdata)):
		xtmp = xdata[idx]
		ytmp = ydata[idx]
		x_tex = precision_tex.format(xtmp)
		if not out_dict.get(x_tex,False):
			out_dict[x_tex] = {"val":ytmp,"idx":idx}
		elif criterion == "min":
			if ytmp < out_dict[x_tex]["val"]:
				out_dict[x_tex]["val"] = ytmp
				out_dict[x_tex]["idx"] = idx
		elif criterion == "max":
			if ytmp > out_dict[x_tex]["val"]:
				out_dict[x_tex]["val"] = ytmp
				out_dict[x_tex]["idx"] = idx
		else:
			sys.exit("ERROR: undefined criterion")
	
	res_list = []
	idx_list = []
	for k,v in out_dict.items():
		res_list.append([float(k),v["val"]])
		idx_list.append([float(k),v["idx"]])
	res_list = sorted(res_list,key=lambda x:x[0])
	idx_list = sorted(idx_list,key=lambda x:x[0])
	return np.array(res_list), np.array(idx_list)

def computeJointEnc(pz,pzcx1,pzcx2,px12):
	px1 = np.sum(px12,axis=1)
	px2 = np.sum(px12,axis=0)
	est_pz_x1 = np.sum(pzcx1*px1[None,:],axis=1)
	est_pz_x2 = np.sum(pzcx2*px2[None,:],axis=1)

	joint_enc= np.zeros((len(pz),len(px1),len(px2)))
	for id1 in range(len(px1)):
		for id2 in range(len(px2)):
			for iz in range(len(pz)):
				joint_enc[iz,id1,id2] = pz[iz] * px1[id1] * px2[id2] / px12[id1,id2] * pzcx1[iz,id1] * pzcx2[iz,id2] / est_pz_x1[iz] / est_pz_x2[iz]
	joint_enc /= np.sum(joint_enc,axis=0,keepdims=True)
	return joint_enc

def expandLogPxcz(log_pxxcz,adim,ndim):
	# return dim (z,x1,x2)
	return np.repeat(np.expand_dims(log_pxxcz.T,axis=adim),repeats=ndim,axis=adim)
def calcProdProb(log_px1cz,log_px2cz,nz):
	expand_log_px1cz = expandLogPxcz(log_px1cz,adim=2,ndim=nx2)
	expand_log_px2cz = expandLogPxcz(log_px2cz,adim=1,ndim=nx1)
	return np.exp(expand_log_px1cz+expand_log_px2cz)/nz
def calcPx1x2(log_px1cz,log_px2cz,nz):
	pzx1x2 = calcProdProb(log_px1cz,log_px2cz,nz)
	return np.sum(pzx1x2,axis=0)
def calcDtvError(log_px1cz,log_px2cz,nz,px1x2):
	pzx1x2 = calcProdProb(log_px1cz,log_px2cz,nz)
	est_px1x2 = np.sum(pzx1x2,axis=0)
	return 0.5 * np.sum(np.fabs(est_px1x2 - px1x2))
def calcProbSoftmax(log_px1cz,log_px2cz):
	expand_log_px1cz = expandLogPxcz(log_px1cz,adim=2,ndim=nx2)
	expand_log_px2cz = expandLogPxcz(log_px2cz,adim=1,ndim=nx1)
	return softmax(expand_log_px1cz+expand_log_px2cz,axis=0)
def calcCondMi(log_px1cz,log_px2cz,nz):
	pzx1x2 = calcProdProb(log_px1cz,log_px2cz,nz)		
	return ut.calcMIcond(np.transpose(pzx1x2,(1,2,0)))