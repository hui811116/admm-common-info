import numpy as np
import sys
import os
import gradient_descent as gd
import utils as ut
import tensorflow as tf
import copy
#import scipy as sp
from scipy.special import softmax

def admmHighDim(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale= kwargs['ss_scale']
	penalty = kwargs['penalty_coeff']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs["pzcx1x2"]
		pz = kwargs["pz"]
		pzcx1 = kwargs["pzcx1"]
		pzcx2 = kwargs["pzcx2"]
		dual_z = kwargs["dual_z"]
		dual_x1 = kwargs["dual_x1"]
		dual_x2 = kwargs["dual_x2"]
	else:
		# random initialization
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# augmented variables
		pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		# dual vars
		dual_z = np.zeros(pz.shape)
		dual_x1= np.zeros(pzcx1.shape)
		dual_x2= np.zeros(pzcx2.shape)
	# gradient masking
	#mask_pzcx1 = np.ones(pzcx1.shape)
	#mask_pzcx2 = np.ones(pzcx2.shape)
	#mask_pzcx1x2 = np.ones(pzcx1x2.shape)
	itcnt = 0
	conv_flag= False
	while itcnt < maxiter:
		itcnt += 1

		err_z = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_x1= np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x2= np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# compute the gradient 

		# NOTE: debugging
		'''
		grad_p = ( (np.log(pzcx1x2)+1) \
				+ (dual_z + penalty * err_z)[:,None,None]\
				+ ((dual_x1 + penalty * err_x1)/px1[None,:])[:,:,None]\
				+np.repeat(np.expand_dims((dual_x2 + penalty * err_x2)/px2[None,:],axis=1),repeats=len(px1),axis=1) )*px1x2[None,:,:]
		'''
		# original
		grad_p = ( (1+gamma) * (np.log(pzcx1x2)+1) \
				+ (dual_z + penalty * err_z)[:,None,None]\
				+ ((dual_x1 + penalty * err_x1)/px1[None,:])[:,:,None]\
				+np.repeat(np.expand_dims((dual_x2 + penalty * err_x2)/px2[None,:],axis=1),repeats=len(px1),axis=1) )*px1x2[None,:,:]

		mean_grad_p = grad_p - np.mean(grad_p,axis=0)
		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		if ss_p == 0:
			#print("pzcx1x2 break")
			#bad_fibers = np.any(pzcx1x2 - mean_grad_p * 1e-9 >= 1,axis=0)
			#mask_pzcx1x2[:,bad_fibers] = 0
			break
		new_pzcx1x2 = pzcx1x2 - mean_grad_p * ss_p
		# new errors
		err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_x1= np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x2= np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2

		# daul updates
		dual_z += penalty * err_z
		dual_x1+= penalty * err_x1
		dual_x2+= penalty * err_x2

		# compute the augmented gradients
		grad_z = -(1-gamma) * (np.log(pz)+1) - dual_z - penalty*err_z
		mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		ss_z = gd.naiveStepSize(pz,-mean_grad_z,ss_init,ss_scale)
		if ss_z == 0:
			break
		grad_x1= -gamma  * (np.log(pzcx1)+1)*px1[None,:] - dual_x1 - penalty * err_x1
		mean_grad_x1 = grad_x1 - np.mean(grad_x1,axis=0)
		ss_x1 = gd.naiveStepSize(pzcx1,-mean_grad_x1,ss_z,ss_scale)
		if ss_x1 == 0:
			break
		grad_x2 = -gamma * (np.log(pzcx2)+1)*px2[None,:] - dual_x2 - penalty * err_x2
		mean_grad_x2 = grad_x2 - np.mean(grad_x2,axis=0)
		ss_x2 = gd.naiveStepSize(pzcx2,-mean_grad_x2,ss_x1,ss_scale)
		if ss_x2 == 0:
			break
		new_pz = pz - mean_grad_z * ss_x2
		new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x2
		new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2
		# new errors
		err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - new_pz
		err_x1= np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - new_pzcx1
		err_x2= np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - new_pzcx2
		# convergence
		conv_z = 0.5 *np.sum(np.abs(err_z))
		conv_x1 = 0.5 *np.sum(np.abs(err_x1),axis=0)
		conv_x2 = 0.5 * np.sum(np.abs(err_x2),axis=0)
		if conv_z< convthres and np.all(conv_x1<convthres) and np.all(conv_x2<convthres):
			conv_flag = True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			pz = new_pz
			pzcx1 = new_pzcx1
			pzcx2 = new_pzcx2
	pz = np.sum(pzcx1x2 * px1x2[None,...],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,...],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,...],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2}

def stoGdDrs(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale= kwargs['ss_scale']
	penalty = kwargs['penalty_coeff']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs["pzcx1x2"]
		q_prob = kwargs["q_prob"]
		dual_p = kwargs["dual_p"]
	else:
		# random initialization
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
		# augmented variables
		q_prob = copy.deepcopy(pzcx1x2)
		# dual vars
		dual_p = np.zeros(pzcx1x2.shape)
	# gradient masking
	itcnt = 0
	conv_flag= False
	while itcnt < maxiter:
		itcnt +=1
		# auxiliary
		err_p = pzcx1x2 - q_prob
		#
		grad_p = (1+gamma)*(np.log(pzcx1x2)+1) * px1x2[None,:,:] + dual_p + penalty * err_p
		mean_grad_p = grad_p - np.mean(grad_p,axis=0,keepdims=True)
		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		if ss_p == 0:
			break
		new_pzcx1x2 = pzcx1x2 - ss_p * mean_grad_p
		# 
		err_p = new_pzcx1x2 - q_prob
		dual_p += penalty * err_p
		# auxiliary
		qz = np.sum(q_prob * px1x2[None,:,:],axis=(1,2))
		qzcx1 = np.sum(q_prob * (px2cx1.T)[None,:,:],axis=2)
		qzcx2 = np.sum(q_prob * px1cx2[None,:,:],axis=1)
		grad_q = -(1-gamma) * (np.log(qz)[:,None,None]+1) *px1x2[None,:,:] \
				 -gamma * (np.log(qzcx1)[:,:,None]+1) * px1x2[None,:,:]\
				 -gamma * (np.repeat(np.expand_dims(np.log(qzcx2),axis=1)+1,repeats=nx1,axis=1)) * px1x2[None,:,:]\
				 -dual_p - penalty* err_p
		mean_grad_q = grad_q - np.mean(grad_q,axis=0,keepdims=True)
		ss_q = gd.naiveStepSize(q_prob,-mean_grad_q,ss_init,ss_scale)
		if ss_q ==0:
			break
		new_q = q_prob - ss_q * mean_grad_q
		#
		err_p = new_pzcx1x2 - new_q
		conv_p = 0.5 * np.sum(np.fabs(err_p),axis=0)
		if np.all(conv_p<convthres):
			conv_flag = True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			q_prob = new_q
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"q_prob":q_prob,"dual_p":dual_p,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2}

'''
def stoLogAdmm(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	#ss_init = kwargs['ss_init']
	#ss_scale= kwargs['ss_scale']
	ss_fixed = kwargs['ss_init']
	penalty = kwargs['penalty_coeff']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs["pzcx1x2"]
		pz = kwargs["pz"]
		pzcx1 = kwargs["pzcx1"]
		pzcx2 = kwargs["pzcx2"]
		dual_z = kwargs["dual_z"]
		dual_x1 = kwargs["dual_x1"]
		dual_x2 = kwargs["dual_x2"]
	else:
		# random initialization
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# augmented variables
		pz = np.sum(pzcx1x2 * px1x2[None,...],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,...],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,...],axis=1)

		# dual vars
		dual_z = np.zeros(pz.shape)
		dual_x1= np.zeros(pzcx1.shape)
		dual_x2= np.zeros(pzcx2.shape)
	# variables
	mlog_pzcx1x2= -np.log(pzcx1x2)
	mlog_pz = -np.log(pz)
	mlog_pzcx1 = -np.log(pzcx1)
	mlog_pzcx2 = -np.log(pzcx2)

	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt +=1 
		# precomputing the error
		exp_mlog_pzcx1x2 = np.exp(-mlog_pzcx1x2)
		exp_mlog_pz = np.exp(-mlog_pz)
		exp_mlog_pzcx1 = np.exp(-mlog_pzcx1)
		exp_mlog_pzcx2 = np.exp(-mlog_pzcx2)
		est_pz = np.sum(exp_mlog_pzcx1x2 * px1x2[None,...],axis=(1,2))
		est_pzcx1 = np.sum(exp_mlog_pzcx1x2 * (px2cx1.T)[None,...],axis=2)
		est_pzcx2 = np.sum(exp_mlog_pzcx1x2 * px1cx2[None,...],axis=1)
		err_z = -np.log(est_pz) - mlog_pz
		err_x1 = -np.log(est_pzcx1) - mlog_pzcx1
		err_x2 = -np.log(est_pzcx2) - mlog_pzcx2
		# gradient of pzcx1x2
		grad_p = -(gamma+1)*exp_mlog_pzcx1x2*((1-mlog_pzcx1x2)*px1x2[None,...]) + \
				(exp_mlog_pzcx1x2 *px1x2[None,...])* np.repeat( np.expand_dims(np.repeat(((dual_z+penalty*err_z)/est_pz)[:,None],repeats=nx1,axis=1),axis=-1),repeats=nx2,axis=-1) +\
				(exp_mlog_pzcx1x2 *(px2cx1.T)[None,...]) * np.repeat( np.expand_dims((dual_x1+penalty*err_x1)/est_pzcx1,axis=-1),repeats=nx2,axis=-1) + \
				(exp_mlog_pzcx1x2 *px1cx2[None,...]) * np.repeat(np.expand_dims((dual_x2+penalty*err_x2)/est_pzcx2,axis=1),repeats=nx1,axis=1)
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - ss_fixed * grad_p
		raw_mlog_pzcx1x2 -= np.amin(raw_mlog_pzcx1x2,axis=0)
		raw_pzcx1x2 = np.exp(-raw_mlog_pzcx1x2) + 1e-9
		new_mlog_pzcx1x2 =-np.log(raw_pzcx1x2 / np.sum(raw_pzcx1x2,axis=0,keepdims=True))

		# update estimates
		exp_mlog_pzcx1x2 = np.exp(-new_mlog_pzcx1x2)
		est_pz = np.sum(exp_mlog_pzcx1x2*px1x2[None,...],axis=(1,2))
		est_pzcx1 = np.sum(exp_mlog_pzcx1x2*(px2cx1.T)[None,...],axis=2)
		est_pzcx2 = np.sum(exp_mlog_pzcx1x2*px1cx2[None,...],axis=1)
		err_z = -np.log(est_pz) - mlog_pz
		err_x1 = -np.log(est_pzcx1) - mlog_pzcx1
		err_x2 = -np.log(est_pzcx2) - mlog_pzcx2

		#dual updates
		dual_z += penalty * err_z 
		dual_x1 += penalty * err_x1
		dual_x2 += penalty* err_x2
		# grad_z
		grad_z = (1-gamma)*exp_mlog_pz *(1-mlog_pz) - (dual_z+penalty*err_z)
		raw_mlog_pz = mlog_pz - ss_fixed * grad_z
		raw_mlog_pz -= np.amin(raw_mlog_pz)
		raw_pz = np.exp(-raw_mlog_pz) + 1e-9
		new_mlog_pz = -np.log(raw_pz/np.sum(raw_pz,keepdims=True))
		# grad_x1
		grad_x1 = gamma * exp_mlog_pzcx1 * (1-mlog_pzcx1) - (dual_x1 + penalty*err_x1)
		raw_mlog_pzcx1 = mlog_pzcx1 - ss_fixed * grad_x1
		raw_mlog_pzcx1 -= np.amin(raw_mlog_pzcx1,axis=0)
		raw_pzcx1 = np.exp(-raw_mlog_pzcx1) + 1e-9
		new_mlog_pzcx1 = -np.log(raw_pzcx1/np.sum(raw_pzcx1,axis=0,keepdims=True))
		# grad_x2
		grad_x2 = gamma * exp_mlog_pzcx2 * (1-mlog_pzcx2) - (dual_x2 + penalty * err_x2)
		raw_mlog_pzcx2 = mlog_pzcx2 - ss_fixed * grad_x2
		raw_mlog_pzcx2 -= np.amin(raw_mlog_pzcx2,axis=0)
		raw_pzcx2 = np.exp(-raw_mlog_pzcx2) + 1e-9
		new_mlog_pzcx2 = -np.log(raw_pzcx2/np.sum(raw_pzcx2,axis=0,keepdims=True))

		# convergence
		conv_z = 0.5 * np.sum(np.fabs(est_pz - np.exp(-new_mlog_pz)))
		conv_x1 = 0.5 * np.sum(np.fabs(est_pzcx1 - np.exp(-new_mlog_pzcx1)),axis=0)
		conv_x2 = 0.5 * np.sum(np.fabs(est_pzcx2 - np.exp(-new_mlog_pzcx2)),axis=0)
		if conv_z < convthres and np.all(conv_x1<convthres) and np.all(conv_x2<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx1x2 = new_mlog_pzcx1x2
			mlog_pz = new_mlog_pz
			mlog_pzcx1 = new_mlog_pzcx1
			mlog_pzcx2 = new_mlog_pzcx2
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,...],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,...],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,...],axis=1)

	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2}
'''

def detComAdmm(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs["ss_init"]
	ss_scale = kwargs["ss_scale"]
	penalty = kwargs["penalty_coeff"]
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	#_debug_alpha = 0
	#_debug_alpha_scale = 1- 1e-4

	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
		pz = kwargs['pz']
		pzcx1 = kwargs['pzcx1']
		pzcx2 = kwargs['pzcx2']
		dual_z = kwargs['dual_z']
		dual_x1 = kwargs['dual_x1']
		dual_x2 = kwargs['dual_x2']
	else:
		# random initialization
		pzcx1x2 = rng.random((nz,nx1,nx2),dtype="float64")
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# precomputed variables
		pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		# dual variables
		dual_z = np.zeros((len(pz),))
		dual_x1 = np.zeros((nz,len(px1),))
		dual_x2 = np.zeros((nz,len(px2),))
	mask_pz = np.ones(pz.shape)
	mask_pzcx1 = np.ones(pzcx1.shape)
	mask_pzcx2 = np.ones(pzcx2.shape)
	mask_pzcx1x2 = np.ones(pzcx1x2.shape)
	conv_flag = False
	itcnt =0
	while itcnt < maxiter:
		itcnt +=1
		# errors
		err_z = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# compute the gradient of the convex part
		# -gamma H(Z|X_1,X_2) + <dual,penalty>
		grad_p = ( gamma * (np.log(pzcx1x2)+1)\
				 + (dual_z+penalty*err_z)[:,None,None]\
				 + ((dual_x1+penalty*err_x1)/px1[None,:])[:,:,None]\
				 + np.repeat(np.expand_dims((dual_x2+penalty*err_x2)/px2[None,:],1),repeats=nx1,axis=1) )*px1x2[None,:,:]
		#print(np.repeat(np.expand_dims((dual_x2+penalty*err_x2)/px2[None,:],1),repeats=nx1,axis=1))
		mean_grad_p = grad_p - np.mean(grad_p,axis=0)

		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p * mask_pzcx1x2,ss_init,ss_scale)
		if ss_p == 0:

			bad_fibers = np.any(pzcx1x2 - mean_grad_p * 1e-7 >= 1.0, axis=0)
			mask_pzcx1x2[:,bad_fibers] = 0
			# pass
		new_pzcx1x2 = pzcx1x2 - ss_p * mean_grad_p * mask_pzcx1x2
		# new error
		err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x2 = np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# dual update
		dual_z += penalty * err_z
		dual_x1 += penalty * err_x1
		dual_x2 += penalty * err_x2

		# grad z, compute (1-gamma)H(Z)
		if not  np.all(pz == pz):
			print(pz)
			sys.exit("bug")
		grad_z = -(1-gamma) * (np.log(pz)+1) - dual_z - penalty * err_z
		mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		# grad x1, compute -gamma H(Z|X_1)
		grad_x1 = -gamma * (np.log(pzcx1)+1) * px1[None,:] - dual_x1 - penalty * err_x1
		mean_grad_x1 = grad_x1 - np.mean(grad_x1,axis=0)
		grad_x2 = -gamma * (np.log(pzcx2)+1) * px2[None,:] - dual_x2 - penalty * err_x2
		mean_grad_x2 = grad_x2 - np.mean(grad_x2,axis=0)
		ss_z = gd.naiveStepSize(pz,-mean_grad_z * mask_pz, ss_init,ss_scale)
		if ss_z == 0:
			if np.any(pz - mean_grad_z * 1e-7 >= 1.0):
				mask_pz = np.zeros(pz.shape)
		ss_x1 = gd.naiveStepSize(pzcx1,-mean_grad_x1 * mask_pzcx1,ss_z,ss_scale)
		if ss_x1 == 0:
			bad_cols = np.any(pzcx1 - mean_grad_x1 * 1e-7 >= 1.0 ,axis=0)
			mask_pzcx1[:,bad_cols] = 0
		ss_x2 = gd.naiveStepSize(pzcx2,-mean_grad_x2 * mask_pzcx2,ss_x1,ss_scale)
		
		if ss_x2 == 0:
			bad_cols = np.any(pzcx2 - mean_grad_x2 * 1e-7 >= 1.0, axis=0)
			mask_pzcx2[:,bad_cols] = 0
		new_pz = pz - mean_grad_z * ss_x2 * mask_pz
		new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x2 * mask_pzcx1
		new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2 * mask_pzcx2

		# error
		err_z = np.sum(new_pzcx1x2* px1x2[None,:,:],axis=(1,2)) - new_pz
		err_x1 = np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=2) - new_pzcx1
		err_x2 = np.sum(new_pzcx1x2 *(px2cx1.T)[None,:,:],axis=1) - new_pzcx2

		# debugging
		conv_z = 0.5 *np.sum(np.abs(err_z * mask_pz))
		conv_x1 = 0.5 *np.sum(np.abs(err_x1 * mask_pzcx1),axis=0)
		conv_x2 = 0.5 * np.sum(np.abs(err_x2 * mask_pzcx2),axis=0)
		if conv_z<convthres and np.all(conv_x1<convthres) and np.all(conv_x2<convthres):
			conv_flag = True
			break
		else:
			# last residuals
			pzcx1x2 = new_pzcx1x2
			pz = new_pz
			pzcx1 = new_pzcx1
			pzcx2 = new_pzcx2
	# NOTE:
	# if need to map to deterministic encoder, then use argmax operation
	# det_pzcx1x2 = np.max(pzcx1x2,axis=0,keepdims=True) == pzcx1x2
	# det_pzcx1x2 = det_pzcx1x2.astype("float32") + 1e-7
	# det_pzcx1x2/= np.sum(det_pzcx1x2,axis=0,keepdims=True)
	return {"conv":conv_flag, "niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2}

'''
# logadmm for deterministic case
def detLogAdmm(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	#ss_init = kwargs["ss_init"]
	#ss_scale = kwargs["ss_scale"]
	ss_fixed = kwargs['ss_init']
	penalty = kwargs["penalty_coeff"]
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px2 = np.sum(px1x2,1)
	px1 = np.sum(px1x2,0)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
		pz = kwargs['pz']
		pzcx1 = kwargs['pzcx1']
		pzcx2 = kwargs['pzcx2']
		dual_z = kwargs['dual_z']
		dual_x1 = kwargs['dual_x1']
		dual_x2 = kwargs['dual_x2']
	else:
		# random initialization
		pzcx1x2 = rng.random((nz,nx1,nx2),dtype="float64")
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# precomputed variables
		pz = np.sum(pzcx1x2 * px1x2[None,...],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,...],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,...],axis=1)

		# dual variables
		dual_z = np.zeros(pz.shape)
		dual_x1 = np.zeros(pzcx1.shape)
		dual_x2 = np.zeros(pzcx2.shape)
	mlog_pzcx1x2 = -np.log(pzcx1x2)
	mlog_pzcx1   = -np.log(pzcx1)
	mlog_pzcx2   = -np.log(pzcx2)
	mlog_pz      = -np.log(pz) 
	itcnt = 0
	conv_flag= False
	while itcnt < maxiter:
		itcnt +=1
		# precomputing error and exponentials
		exp_mlog_pz = np.exp(-mlog_pz)
		exp_mlog_pzcx1 = np.exp(-mlog_pzcx1)
		exp_mlog_pzcx2 = np.exp(-mlog_pzcx2)
		exp_mlog_pzcx1x2 = np.exp(-mlog_pzcx1x2)
		est_pzcx1 = np.sum(exp_mlog_pzcx1x2 * (px2cx1.T)[None,...],axis=2)
		est_pzcx2 = np.sum(exp_mlog_pzcx1x2 * px1cx2[None,...],axis=1)
		est_pz = np.sum(exp_mlog_pzcx1x2 * px1x2[None,...],axis=(1,2))

		# errors
		err_z = -np.log(est_pz)-mlog_pz
		err_x1 = -np.log(est_pzcx1)-mlog_pzcx1
		err_x2 = -np.log(est_pzcx2)-mlog_pzcx2

		# gradient of z|x1x2
		grad_p = -gamma*exp_mlog_pzcx1x2*(1-mlog_pzcx1x2)*px1x2[None,...] \
				+(exp_mlog_pzcx1x2 * (px2cx1.T)[None,...])*(( dual_x1 + penalty * err_x1)[...,None]/est_pzcx1[...,None]) \
				+(exp_mlog_pzcx1x2 * px1cx2[None,...])*np.repeat(np.expand_dims((dual_x2 + penalty* err_x2) /est_pzcx2,axis=1),repeats=nx1,axis=1)\
				+(exp_mlog_pzcx1x2*px1x2[None,...])*np.repeat(np.expand_dims(np.repeat(((dual_z + penalty* err_z) /est_pz)[:,None],repeats=nx1,axis=1),axis=2),repeats=nx2,axis=2)
		# projection, pzcx1x2
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - grad_p * ss_fixed
		raw_mlog_pzcx1x2 -= np.amin(raw_mlog_pzcx1x2,axis=0)
		raw_pzcx1x2 = np.exp(-raw_mlog_pzcx1x2) + 1e-9
		new_mlog_pzcx1x2 = -np.log(raw_pzcx1x2/np.sum(raw_pzcx1x2,axis=0,keepdims=True))

		# update temp vars
		exp_mlog_pzcx1x2 = np.exp(-new_mlog_pzcx1x2)
		est_pzcx1 = np.sum(exp_mlog_pzcx1x2 * (px2cx1.T)[None,...],axis=2)
		est_pzcx2 = np.sum(exp_mlog_pzcx1x2 * px1cx2[None,...],axis=1)
		est_pz = np.sum(exp_mlog_pzcx1x2 * px1x2[None,...],axis=(1,2))

		# update errors
		err_z = -np.log(est_pz)-mlog_pz
		err_x1 = -np.log(est_pzcx1)-mlog_pzcx1
		err_x2 = -np.log(est_pzcx2)-mlog_pzcx2

		# update dual variables
		dual_z += penalty * err_z
		dual_x1 += penalty * err_x1
		dual_x2 += penalty * err_x2

		# gradient of z
		grad_z = (1-gamma) * exp_mlog_pz*(1-mlog_pz) - (dual_z+penalty * err_z)
		# z projection
		raw_mlog_pz = mlog_pz - grad_z * ss_fixed
		raw_mlog_pz -= np.amin(raw_mlog_pz)
		raw_pz = np.exp(-raw_mlog_pz) + 1e-9
		new_pz = raw_pz/np.sum(raw_pz)
		new_mlog_pz = -np.log(new_pz)

		# gradient of x1
		grad_x1 = gamma * exp_mlog_pzcx1*(1-mlog_pzcx1) - (dual_x1 + penalty * err_x1)
		# x1 projection
		raw_mlog_pzcx1 = mlog_pzcx1 - grad_x1 * ss_fixed
		raw_mlog_pzcx1 -= np.amin(raw_mlog_pzcx1,axis=0)
		raw_pzcx1 = np.exp(-raw_mlog_pzcx1) + 1e-9
		new_pzcx1 = raw_pzcx1/np.sum(raw_pzcx1,axis=0,keepdims=True)
		new_mlog_pzcx1 = -np.log(new_pzcx1)

		# gradient of x2
		grad_x2 = gamma * exp_mlog_pzcx2*(1-mlog_pzcx2) - (dual_x2 + penalty*err_x2)
		# x2 projection
		raw_mlog_pzcx2 = mlog_pzcx2 = grad_x2 * ss_fixed
		raw_mlog_pzcx2 -= np.amin(raw_mlog_pzcx2,axis=0)
		raw_pzcx2 = np.exp(-raw_mlog_pzcx2) + 1e-9
		new_pzcx2 = raw_pzcx2 / np.sum(raw_pzcx2,axis=0,keepdims=True)
		new_mlog_pzcx2 = -np.log(new_pzcx2)

		# convergence # back to probability
		conv_z = 0.5 * np.sum(np.fabs(est_pz-new_pz))
		conv_x1 = 0.5 * np.sum(np.fabs(est_pzcx1-new_pzcx1),axis=0)
		conv_x2 = 0.5 * np.sum(np.fabs(est_pzcx2-new_pzcx2),axis=0)
		if np.all(conv_z < convthres) and np.all(conv_x1<convthres) and np.all(conv_x2 < convthres):
			conv_flag = True
			break
		else:
			mlog_pz = new_mlog_pz
			mlog_pzcx1 = new_mlog_pzcx1
			mlog_pzcx2 = new_mlog_pzcx2
			mlog_pzcx1x2 = new_mlog_pzcx1x2
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,...],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,...],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,...],axis=1)
	return {"conv":conv_flag,"niter":itcnt,
			"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,
			"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2}
'''
# compared algorithms
# gradient based methods
# Sula, E.; Gastpar, M.C. Common Information Components Analysis. Entropy 2021, 23, 151.
# https://doi.org/10.3390/e23020151

# in its relaxed Wyner common information, the gradients are taken w.r.t. the (conditional) mutual information
# which can be equivalently expressed as derivative to a combination of entropy and conditional entropy functions.

def stoGradComp(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale = kwargs['ss_scale']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

	#mask_pzcx1x2 = np.ones(pzcx1x2.shape)
	itcnt =0 
	conv_flag = False
	while itcnt < maxiter:
		itcnt += 1
		# auxiliary variables
		pz = np.sum(pzcx1x2*px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
		# calculate the gradient
		grad_p = ((1+gamma) * (np.log(pzcx1x2)+1)) * px1x2[None,:,:]\
					-(1-gamma) *( (np.log(pz) + 1)[:,None,None] * px1x2[None,:,:])\
					-gamma * (np.log(pzcx1) +1)[:,:,None] * px1x2[None,:,:] \
					-gamma * (np.log(np.repeat(np.expand_dims(pzcx2,axis=1),repeats=nx1,axis=1))+1)*px1x2[None,:,:]
		mean_grad_p = grad_p - np.mean(grad_p,axis=0,keepdims=True)
		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		if ss_p == 0:
			break
		new_pzcx1x2 = pzcx1x2 - mean_grad_p * ss_p

		# the compared method project the obtained encoder to the wyner setting
		#new_pz = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		#ent_z = -np.sum(new_pz*np.log(new_pz))
		#ent_pzcx1x2 = -np.sum(new_pzcx1x2 * px1x2[None,:,:] * np.log(new_pzcx1x2))
		# the convergence criterion of the reference uses
		conv_z = 0.5 * np.sum(np.fabs(new_pzcx1x2 - pzcx1x2),axis=0) # total variation
		if np.all(conv_z<convthres):
			conv_flag=True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	pz = np.sum(pzcx1x2*px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2}
'''
def gdCompTwoStep(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale = kwargs['ss_scale']
	# extra convergence criterion
	miC_thres = 1e-3
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	itcnt =0
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		# auxiliary variables
		pz = np.sum(pzcx1x2*px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
		# convergence metrics
		entz = np.sum(-pz*np.log(pz))
		entzcx1x2 = np.sum(-pzcx1x2*px1x2[None,:,:]*np.log(pzcx1x2))
		mi_C = entz - entzcx1x2
		# gradient
		# C= I(X1,X2;Z) = H(Z) - H(Z|X1,X2)
		grad_C = -px1x2[None,:,:]*(np.log(pz)+1)[:,None,None]\
				+ px1x2[None,:,:]*(np.log(pzcx1x2)+1)
		# J = I(X1;X2|Z) = I(X1,X2;Z) - I(X1;Z) - I(X2;Z) + I(X1,X2)
		#   = -H(Z)-H(Z|X1,X2)+H(Z|X1)+H(Z|X2)
		grad_J = px1x2[None,:,:]*(np.log(pz)+1)[:,None,None]\
				+px1x2[None,:,:]*(np.log(pzcx1x2)+1)\
				-px1x2[None,:,:]*(np.log(pzcx1)[:,:,None]+1)\
				-px1x2[None,:,:]*(np.expand_dims(np.log(pzcx2),axis=1)+1)

		grad_sum = grad_C + gamma * grad_J
		mean_grad_sum = grad_sum - np.mean(grad_sum,axis=0)
		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_sum,ss_init,ss_scale)
		if ss_p ==0:
			break
		new_pzcx1x2 = pzcx1x2 - mean_grad_sum*ss_p
		# convergence
		new_pz = np.sum(new_pzcx1x2*px1x2[None,:,:],axis=(1,2))
		new_entz = np.sum(-new_pz * np.log(new_pz))
		new_pzcx1 = np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=2)
		new_pzcx2 = np.sum(new_pzcx1x2 * np.expand_dims(px2cx1,axis=1),axis=1)
		new_entzcx1x2 = np.sum(-new_pzcx1x2 * px1x2[None,:,:] * np.log(new_pzcx1x2))
		new_mi_C = new_entz - new_entzcx1x2
		diff_mi_C = np.fabs(new_mi_C - mi_C)
		if diff_mi_C<miC_thres:
			break
		else:
			pzcx1x2 = new_pzcx1x2

	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2}
'''
'''
def stoLogGrad(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_fixed = kwargs['ss_init']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	mlog_pzcx1x2 = -np.log(pzcx1x2)
	itcnt =0
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		# auxiliary variables
		pzcx1x2 = np.exp(-mlog_pzcx1x2)
		pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		grad_p = -(1+gamma)*pzcx1x2 * px1x2[None,:,:] * (1-mlog_pzcx1x2) \
				 +gamma*pzcx1x2*px1x2[None,:,:]*(1+np.log(pzcx1)[:,:,None]) \
				 +gamma*pzcx1x2*px1x2[None,:,:]*(1+np.expand_dims(np.log(pzcx2),axis=1)) \
				 +(1-gamma)*pzcx1x2*px1x2[None,:,:]*(1+np.log(pz)[:,None,None])
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - ss_fixed * grad_p
		min_mlog_pzcx1x2 = np.amin(raw_mlog_pzcx1x2,axis=0)
		min_mlog_pzcx1x2 = np.where(min_mlog_pzcx1x2<0.0,min_mlog_pzcx1x2,np.zeros((nx1,nx2)))
		fixed_mlog_pzcx1x2 = raw_mlog_pzcx1x2 - min_mlog_pzcx1x2[None,:,:]
		raw_pzcx1x2 = np.exp(-fixed_mlog_pzcx1x2) + 1e-9
		new_pzcx1x2 = raw_pzcx1x2 / np.sum(raw_pzcx1x2,axis=0,keepdims=0)
		new_mlog_pzcx1x2 = -np.log(new_pzcx1x2)

		conv_z = np.sum(np.fabs(new_pzcx1x2 - pzcx1x2),axis=0)
		if np.all(conv_z<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx1x2 = new_mlog_pzcx1x2
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,'pzcx1':pzcx1,"pzcx2":pzcx2}
'''
def stoLogDRS(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_fixed = kwargs['ss_init']
	d_seed = None
	penalty = kwargs['penalty_coeff']
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
		dual_p = kwargs['dual_p']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	mlog_pzcx1x2 = -np.log(pzcx1x2)
	mlog_q = copy.deepcopy(mlog_pzcx1x2)
	dual_p = np.zeros((nz,nx1,nx2))

	itcnt =0
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		# p update
		# auxiliary variables for F
		aux_p = np.exp(-mlog_pzcx1x2)
		aux_pz = np.sum(aux_p * px1x2[None,:,:],axis=(1,2))
		#aux_pzcx1 = np.sum(aux_p * (px2cx1.T)[None,:,:],axis=2)
		#aux_pzcx2 = np.sum(aux_p * px1cx2[None,:,:],axis=1)
		err_p = mlog_pzcx1x2 - mlog_q

		grad_p = -(1+gamma) * aux_p * px1x2[None,:,:] * (1- mlog_pzcx1x2) \
				 +(1-gamma) * aux_p * px1x2[None,:,:] * (1+np.log(aux_pz)[:,None,None]) \
				 + dual_p + penalty * err_p
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - grad_p * ss_fixed
		# projection
		min_p = np.amin(raw_mlog_pzcx1x2,axis=0)
		min_p = np.where(min_p<0.0,min_p,np.zeros((nx1,nx2)))
		prj_mlog_pzcx1x2 = raw_mlog_pzcx1x2 - min_p
		raw_pzcx1x2 = np.exp(-prj_mlog_pzcx1x2) + 1e-9
		new_pzcx1x2 = raw_pzcx1x2 / np.sum(raw_pzcx1x2,axis=0,keepdims=True)
		new_mlog_pzcx1x2 = -np.log(new_pzcx1x2)

		# dual update
		err_p = new_mlog_pzcx1x2 - mlog_q
		dual_p += penalty * err_p

		# auxiliary variables for G
		q_prob = np.exp(-mlog_q)
		q_pzcx1 = np.sum(q_prob * (px2cx1.T)[None,:,:],axis=2)
		q_pzcx2 = np.sum(q_prob * px1cx2[None,:,:],axis=1)

		# FIXME: can be optimized for faster runtime, but keep as its separate gradients for clearness
		grad_q = gamma * q_prob * px1x2[None,:,:] * (1+np.log(q_pzcx1)[:,:,None]) \
				+gamma * q_prob * px1x2[None,:,:] * (1+np.expand_dims(np.log(q_pzcx2),axis=1)) \
				-dual_p - penalty * err_p

		raw_mlog_q = mlog_q - grad_q * ss_fixed
		min_q = np.amin(raw_mlog_q,axis=0)
		min_q = np.where(min_q<0,min_q,np.zeros((nx1,nx2)))
		prj_mlog_q = raw_mlog_q - min_q
		raw_q = np.exp(-prj_mlog_q) + 1e-9
		new_q = raw_q / np.sum(raw_q,axis=0,keepdims=True)
		new_mlog_q = -np.log(new_q)

		# convergence 
		#err_p = new_mlog_pzcx1x2 - new_mlog_q
		conv_p = np.sum(np.fabs(new_pzcx1x2-new_q),axis=0)
		if np.all(conv_p<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx1x2 = new_mlog_pzcx1x2
			mlog_q = new_mlog_q
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,'pzcx1':pzcx1,"pzcx2":pzcx2,"dual_p":dual_p}

def stoLogDrsVar(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_fixed = kwargs['ss_init']
	d_seed = None
	penalty = kwargs['penalty_coeff']
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
		dual_p = kwargs['dual_p']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	mlog_pzcx1x2 = -np.log(pzcx1x2)
	mlog_q = copy.deepcopy(mlog_pzcx1x2)
	dual_p = np.zeros((nz,nx1,nx2))

	itcnt =0
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		aux_p = np.exp(-mlog_pzcx1x2)
		err_p = mlog_pzcx1x2 - mlog_q
		#
		grad_p = -(1+gamma) * aux_p * px1x2[None,:,:] * (1-mlog_pzcx1x2) + dual_p + penalty * err_p
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - grad_p * ss_fixed
		min_p = np.amin(raw_mlog_pzcx1x2,axis=0)
		min_p = np.where(min_p<0.0,min_p,np.zeros((nx1,nx2)))
		prj_mlog_p = raw_mlog_pzcx1x2 - min_p[None,:,:]
		raw_pzcx1x2 = np.exp(-prj_mlog_p) + 1e-9
		new_pzcx1x2 = raw_pzcx1x2 / np.sum(raw_pzcx1x2,axis=0,keepdims=True)
		new_mlog_pzcx1x2 = -np.log(new_pzcx1x2)
		# 
		err_p = new_mlog_pzcx1x2 - mlog_q
		#
		dual_p += penalty * err_p
		#
		aux_q = np.exp(-mlog_q)
		aux_qz = np.sum(aux_q * px1x2[None,:,:],axis=(1,2))
		aux_qzcx1 = np.sum(aux_q * (px2cx1.T)[None,:,:],axis=2)
		aux_qzcx2 = np.sum(aux_q * px1cx2[None,:,:],axis=1)
		
		grad_q = (1-gamma)*aux_q * px1x2[None,:,:] * (1+np.log(aux_qz)[:,None,None])\
				+gamma * aux_q * px1x2[None,:,:] * (1+np.log(aux_qzcx1)[:,:,None])\
				+gamma * aux_q * px1x2[None,:,:] * (1+np.expand_dims(np.log(aux_qzcx2),axis=1))\
				-dual_p - penalty * err_p
		raw_mlog_q = mlog_q - grad_q * ss_fixed
		min_q = np.amin(raw_mlog_q,axis=0)
		min_q = np.where(min_q<0.0,min_q,np.zeros((nx1,nx2)))
		prj_mlog_q = raw_mlog_q - min_q[None,:,:]
		raw_q = np.exp(-prj_mlog_q) + 1e-9
		new_q = raw_q / np.sum(raw_q,axis=0,keepdims=True)
		new_mlog_q = -np.log(new_q)
		#
		err_p = new_mlog_pzcx1x2 - new_mlog_q
		dtv_p = 0.5 * np.sum(np.fabs(new_pzcx1x2-new_q),axis=0)
		if np.all(dtv_p<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx1x2 = new_mlog_pzcx1x2
			mlog_q = new_mlog_q
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,'pzcx1':pzcx1,"pzcx2":pzcx2,"dual_p":dual_p}

def detLogDrs(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_fixed = kwargs['ss_init']
	d_seed = None
	penalty = kwargs['penalty_coeff']
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
		dual_p = kwargs['dual_p']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	mlog_pzcx1x2 = -np.log(pzcx1x2)
	mlog_q = copy.deepcopy(mlog_pzcx1x2)
	dual_p = np.zeros((nz,nx1,nx2))

	itcnt =0
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		# exact common information
		# loss = H(Z) + \gamma (I(Z;X1,X2) - I(Z;X1) - I(Z;X2) + I(X1;X2))
		# equivalent loss = (1-gamma)H(Z) -gamma H(Z|X1,X2) + gamma* H(Z|X1) + gamma* H(Z|X2)
		aux_p = np.exp(-mlog_pzcx1x2)
		err_p = mlog_pzcx1x2 - mlog_q
		# penalty <nu, mlogp - mlogq> + 0.5c * |mlogp - mlogq|^2
		grad_p = -gamma * aux_p * (1-mlog_pzcx1x2) * px1x2[None,:,:] + dual_p + err_p * penalty
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - ss_fixed * grad_p
		# projection
		min_p = np.amin(raw_mlog_pzcx1x2,axis=0)
		min_p = np.where(min_p<0.0,min_p,np.zeros((nx1,nx2)))
		prj_mlog_p = raw_mlog_pzcx1x2 - min_p[None,:,:]
		raw_pzcx1x2 = np.exp(-prj_mlog_p) + 1e-9
		new_p = raw_pzcx1x2 / np.sum(raw_pzcx1x2,axis=0,keepdims=True)
		new_mlog_pzcx1x2 = -np.log(new_p)

		err_p = new_mlog_pzcx1x2 - mlog_q
		# dual ascend
		dual_p += penalty * err_p
		# q update
		aux_q = np.exp(-mlog_q)
		aux_qz = np.sum(aux_q * px1x2[None,:,:],axis=(1,2))
		aux_pzcx1 = np.sum(aux_q* (px2cx1.T)[None,:,:],axis=2)
		aux_pzcx2 = np.sum(aux_q* px1cx2[None,:,:],axis=1)
		grad_q = (1-gamma)* aux_q* px1x2[None,:,:] * (1+np.log(aux_qz)[:,None,None]) \
				 + gamma  * aux_q* px1x2[None,:,:] * (1+np.log(aux_pzcx1)[:,:,None]) \
				 + gamma  * aux_q* px1x2[None,:,:] * (1+np.expand_dims(np.log(aux_pzcx2),axis=1))\
				 - dual_p - penalty * err_p
		# projection for q
		raw_mlog_q = mlog_q - ss_fixed * grad_q
		min_q = np.amin(raw_mlog_q,axis=0)
		min_q = np.where(min_q<0.0,min_q,np.zeros((nx1,nx2)))
		prj_mlog_q = raw_mlog_q - min_q[None,:,:]
		raw_q = np.exp(-prj_mlog_q) + 1e-9
		new_q = raw_q/np.sum(raw_q,axis=0,keepdims=True)
		new_mlog_q = -np.log(new_q)

		# convergence
		#err_p = new_mlog_pzcx1x2 - new_mlog_q
		# total variation
		dtv_p = 0.5 * np.sum(np.fabs(new_p-new_q),axis=0)
		#conv_p = np.sum(np.fabs(err_p),axis=0)
		if np.all(dtv_p<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx1x2 = new_mlog_pzcx1x2
			mlog_q = new_mlog_q
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"dual_p":dual_p,}

# dev
# scalable splitting method based wyner common information solver.
# 
def wynerDrs(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_fixed = kwargs['ss_init']
	d_seed = None
	penalty = kwargs['penalty_coeff']
	penalty_max = kwargs['penalty_coeff']
	#penalty = 1.0
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,1)
	px2 = np.sum(px1x2,0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		#pzcx1x2 = kwargs['pzcx1x2']
		px1cz = kwargs['px1cz']
		px2cz = kwargs['px2cz']
	else:
		px1cz = rng.random((nx1,nz))
		px1cz /= np.sum(px1cz,axis=0,keepdims=True)
		px2cz = rng.random((nx2,nz))
		px2cz /= np.sum(px2cz,axis=0,keepdims=True)
		# NOTE: Assume p(z) is uniformly distributed 
		# initial pzcx1x2 is the softmax of the product px1cz, px2cz
	def softmaxFromLog(log_px1cz,log_px2cz):
		expand_log_px1cz_T = np.repeat(np.expand_dims(log_px1cz.T,axis=2),repeats=nx2,axis=2)
		expand_log_px2cz_T = np.repeat(np.expand_dims(log_px2cz.T,axis=1),repeats=nx1,axis=1)
		return softmax(expand_log_px1cz_T + expand_log_px2cz_T, axis=0)
	def probNumerator(log_px1cz,log_px2cz,nz):
		expand_log_px1cz_T = np.repeat(np.expand_dims(log_px1cz.T,axis=2),repeats=nx2,axis=2)
		expand_log_px2cz_T = np.repeat(np.expand_dims(log_px2cz.T,axis=1),repeats=nx1,axis=1)
		return np.exp(expand_log_px1cz_T + expand_log_px2cz_T)/nz
	log_px1cz = np.log(px1cz)
	log_px2cz = np.log(px2cz)
	pzcx1x2 = softmaxFromLog(log_px1cz,log_px2cz)
	# dual...
	dual_v = np.zeros((nx1,nx2))
	itcnt =0 
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		# check function value for debugging and performance
		pzcx1x2 = softmaxFromLog(log_px1cz,log_px2cz)
		prob_num = probNumerator(log_px1cz,log_px2cz,nz)
		est_px1x2 = np.sum(prob_num,axis=0)
		#print("error",0.5*np.sum(np.fabs(est_px1x2 - px1x2)))
		#print(np.sum(pzcx1x2 * est_px1x2[None,:,:]))
		grad_x1 = (1/nz) * (1+log_px1cz)*np.exp(log_px1cz)
		grad_x1 += penalty * np.sum(prob_num*(np.log(prob_num)-np.log(px1x2)[None,:,:]+1),axis=2).T
		# projection onto the negative real line
		raw_log_px1cz = log_px1cz - ss_fixed * grad_x1
		max_x1 = np.amax(raw_log_px1cz,axis=0)
		max_x1 = np.where(max_x1>0.0,max_x1,np.zeros((nz)))
		raw_log_px1cz -= max_x1[None,:]
		raw_px1cz = np.exp(raw_log_px1cz) + 1e-9 # smoothing
		new_px1cz = raw_px1cz / np.sum(raw_px1cz,axis=0,keepdims=True)
		new_log_px1cz = np.log(new_px1cz)

		prob_num = probNumerator(new_log_px1cz,log_px2cz,nz)
		grad_x2 = (1/nz) * (1+log_px2cz)*np.exp(log_px2cz)
		grad_x2 += penalty * np.sum(prob_num*(np.log(prob_num)-np.log(px1x2)[None,:,:]+1),axis=1).T

		raw_log_px2cz = log_px2cz - ss_fixed * grad_x2
		max_x2 = np.amax(raw_log_px2cz,axis=0)
		max_x2 = np.where(max_x2>0.0,max_x2,np.zeros((nz)))
		raw_log_px2cz -= max_x2[None,:]
		raw_px2cz = np.exp(raw_log_px2cz) + 1e-9
		new_px2cz = raw_px2cz / np.sum(raw_px2cz,axis=0,keepdims=True)
		new_log_px2cz = np.log(new_px2cz)
		#
		prob_num = probNumerator(new_log_px1cz,new_log_px2cz,nz)
		
		err_x12 = np.sum(prob_num,axis=0) - px1x2 

		# convergence
		dtv_x1x2 = 0.5 * np.sum(np.fabs(err_x12))
		if itcnt % 10000 ==0:
			print(np.sum(prob_num,axis=0))
			#print(grad_x1)
			print("iteration:{:}, error:{:}".format(itcnt,dtv_x1x2))
		if dtv_x1x2<convthres:
			conv_flag = True
			break
		else:
			penalty *= 1.1
			penalty = min(penalty,penalty_max)
			log_px1cz = new_log_px1cz
			log_px2cz = new_log_px2cz

	pzcx1x2 =softmaxFromLog(log_px1cz,log_px2cz)
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"dual_v":dual_v,}
