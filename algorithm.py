import numpy as np
import sys
import os
import gradient_descent as gd
import utils as ut
import tensorflow as tf
#import scipy as sp
#from scipy.special import softmax

def admmHighDim(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale= kwargs['ss_scale']
	penalty = kwargs['penalty_coeff']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px2 = np.sum(px1x2,0)
	px1 = np.sum(px1x2,1)
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
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2}

'''
def egSolver(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px2 = np.sum(px1x2,0)
	px1 = np.sum(px1x2,1)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	# random initialization
	pzcx1x2 = rng.random((nz,nx1,nx2))
	pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

	# precomputed variables
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

	itcnt = 0
	conv_flag = False
	while itcnt < maxiter:
		itcnt += 1
		# compute the raw probabilities
		ker_z = ((1-gamma)/(1+gamma) * np.log(pz)[:,None] + gamma/(1+gamma) * np.log(pzcx1))[:,:,None]\
				 + np.repeat(np.expand_dims(gamma/(1+gamma) * np.log(pzcx2),1),repeats=len(px1),axis=1)
		new_pzcx1x2 = np.exp(ker_z)
		new_pzcx1x2 /= np.sum(new_pzcx1x2,axis=0)
		# update
		pz = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1)
		# convergence
		conv = 0.5 * np.sum(np.abs(new_pzcx1x2 - pzcx1x2),axis=0)
		if np.all(conv < convthres):
			conv_flag = True
			break
		else:
			# update probabilities
			pzcx1x2 = new_pzcx1x2

	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2}
'''
'''
def distAdmmSolver(px1x2,nz,gamma1,gamma2,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale = kwargs['ss_scale']
	penalty = kwargs['penalty_coeff']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px2 = np.sum(px1x2,0)
	px1 = np.sum(px1x2,1)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T
	# INITIALIZATION
	# depends on the flag
	# random initialization
	if "load" in kwargs.keys():
		# load from precomputed solution
		pzcx1 = kwargs["pzcx1"]
		pzcx2 = kwargs["pzcx2"]
		pz    = kwargs["pz"]
		# dual variables need to be loaded too
		dual_x1 = kwargs["dual_x1"]
		dual_x2 = kwargs["dual_x2"]

	else:
		# use random initialization instead
		pzcx1 = rng.random((nz,len(px1)))
		pzcx1 /= np.sum(pzcx1,0,keepdims=True)
		pzcx2 = rng.random((nz,len(px2)))
		pzcx2 /= np.sum(pzcx2,0,keepdims=True)
		pz = (np.sum(pzcx1 * px1[None,:],axis=1) + np.sum(pzcx2 * px2[None,:],axis=1))/2
		pz /= np.sum(pz)
		# dual
		dual_x1 = pz - np.sum(pzcx1 * px1[None,:],axis=1)
		dual_x2 = pz - np.sum(pzcx2 * px2[None,:],axis=1)

	itcnt = 0
	conv_flag = False
	while itcnt < maxiter:
		itcnt += 1
		# precompute 
		tmp_px1cz = (pzcx1 * px1[None,:] / np.sum(pzcx1 * px1[None,:],axis=1,keepdims=True)).T
		tmp_px2cz = (pzcx2 * px2[None,:] / np.sum(pzcx2 * px2[None,:],axis=1,keepdims=True)).T
		ep_x1cz = np.repeat(np.expand_dims(tmp_px1cz,1),len(px2),1)
		ep_x2cz = np.repeat(np.expand_dims(tmp_px2cz,0),len(px1),0)
		tmp_joint_z = ep_x1cz * ep_x2cz # (x1,x2,z)
		# err
		err_x1 = pz - np.sum(pzcx1 * px1[None,:],axis=1)
		err_x2 = pz - np.sum(pzcx2 * px2[None,:],axis=1)
		# pz step
		# compute joint dkl
		joint_dkl = np.sum(tmp_joint_z * (np.log(tmp_joint_z)-np.log(px1x2)[:,:,None]),axis=(0,1))
		grad_z = -1 * (2-gamma1-gamma2) * np.log(pz)-(2-gamma1-gamma2) - joint_dkl\
				 + dual_x1 + dual_x2 + penalty * (err_x1 + err_x2)
		mean_grad_z = grad_z - np.mean(grad_z)
		ss_z = gd.naiveStepSize(pz,-mean_grad_z,ss_init,ss_scale)
		if ss_z == 0:
			break
		new_pz = pz - ss_z * mean_grad_z
		# update err and dual
		err_x1 = new_pz - np.sum(pzcx1 * px1[None,:],axis=1)
		err_x2 = new_pz - np.sum(pzcx2 * px2[None,:],axis=1)
		dual_x1 += penalty * err_x1
		dual_x2 += penalty * err_x2

		# compute the kl divergecnes
		tmp_dkl_cx2z = np.sum(ep_x1cz * (np.log(ep_x1cz) -np.log(px1cx2)[:,:,None]),axis=0)
		tmp_dkl_cx1z = np.sum(ep_x2cz * (np.log(ep_x2cz) -np.log(px2cx1.T)[:,:,None]),axis=1)
		# encoder update
		grad_x1 = ((1-gamma1)*np.log(pzcx1)+tmp_dkl_cx1z.T+(1-gamma1)- (dual_x1 + penalty*err_x1)[:,None]  )*px1[None,:]
		mean_grad_x1 = grad_x1 - np.mean(grad_x1,0)
		ss_x1 = gd.naiveStepSize(pzcx1,-mean_grad_x1,ss_init,ss_scale)
		if ss_x1 == 0:
			break
		new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x1

		grad_x2 = ((1-gamma2)*np.log(pzcx2)+tmp_dkl_cx2z.T+(1-gamma2)- (dual_x2 +penalty*err_x2)[:,None] )*px2[None,:]
		mean_grad_x2 = grad_x2 - np.mean(grad_x2,0)
		ss_x2 = gd.naiveStepSize(pzcx2,-mean_grad_x2,ss_init,ss_scale)
		if ss_x2 == 0:
			break
		new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2
		# error
		err_x1 = new_pz - np.sum(new_pzcx1 * px1[None,:],axis=1)
		err_x2 = new_pz - np.sum(new_pzcx2 * px2[None,:],axis=1)
		conv_x1 = 0.5 * np.sum(np.abs(err_x1)) # total variation
		conv_x2 = 0.5 * np.sum(np.abs(err_x2))
		tmp_pzcx1x2 = ut.computeJointEnc(pz,pzcx1,pzcx2,px1x2)
		err_zcx1 = new_pzcx1 - np.sum(tmp_pzcx1x2 * px1cx2[None,:,:],axis=2)
		err_zcx2 = new_pzcx2 - np.sum(tmp_pzcx1x2 * (px2cx1.T)[None,:,:],axis=1)
		conv_zcx1 = np.all(np.abs(err_zcx1)<convthres)
		conv_zcx2 = np.all(np.abs(err_zcx2)<convthres)
		if conv_x1 < convthres and conv_x2 < convthres and conv_zcx1 and conv_zcx2:
			conv_flag = True
			break
		else:
			pzcx1 = new_pzcx1
			pzcx2 = new_pzcx2 
			pz = new_pz
	# approximated pzcx1x2
	pzcx1x2 = ut.computeJointEnc(pz,pzcx1,pzcx2,px1x2)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1":pzcx1,"pzcx2":pzcx2,"pz":pz,"pzcx1x2":pzcx1x2,"dual_x1":dual_x1,"dual_x2":dual_x2}
'''


'''
# WARNING: this is not a efficient algorithm (when |X| grows)
#          because it simply projects a random point to the closest feasible point
def expgdDetCom(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px2 = np.sum(px1x2,0)
	px1 = np.sum(px1x2,1)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	# random initialization
	pzcx1x2 = rng.random((nz,nx1,nx2))
	pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

	# precomputed variables
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt += 1

		# compute the raw kernal
		ker_z = (1/gamma -1 ) * np.log(pz)[:,None,None] + np.log(pzcx1)[:,:,None] + np.repeat(np.expand_dims(np.log(pzcx2),1),repeats=len(px1),axis=1)
		# forcing to deterministic by using argmax
		new_pzcx1x2 = ker_z == np.amax(ker_z,axis=0,keepdims=True)
		new_pzcx1x2 = new_pzcx1x2.astype("float32") + 1e-7
		new_pzcx1x2/=np.sum(new_pzcx1x2,axis=0)
		# 
		pz = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=2)
		pzcx2 = np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=1)
		conv = 0.5 * np.sum(np.abs(new_pzcx1x2 - pzcx1x2),axis=0)
		if np.all(conv < convthres):
			conv_flag = True
			break
		else:
			pzcx1x2 = new_pzcx1x2
	return {"conv":conv_flag,"niter":itcnt,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"pzcx1x2":pzcx1x2}
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
	px2 = np.sum(px1x2,0)
	px1 = np.sum(px1x2,1)
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
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# precomputed variables
		pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		# dual variables
		dual_z = np.zeros((len(pz),))
		dual_x1 = np.zeros((nz,len(px1),))
		dual_x2 = np.zeros((nz,len(px2),))
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
		grad_p = (gamma * (np.log(pzcx1x2)+1)\
				 + (dual_z+penalty*err_z)[:,None,None]\
				 + ((dual_x1+penalty*err_x1)/px1[None,:])[:,:,None]\
				 + np.repeat(np.expand_dims((dual_x2+penalty*err_x2)/px2[None,:],1),repeats=nx1,axis=1) )*px1x2[None,:,:]
		#print(np.repeat(np.expand_dims((dual_x2+penalty*err_x2)/px2[None,:],1),repeats=nx1,axis=1))
		mean_grad_p = grad_p - np.mean(grad_p,axis=0)

		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		if ss_p == 0:
			break
		new_pzcx1x2 = pzcx1x2 - ss_p * mean_grad_p

		# new error
		err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x2 = np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# dual update
		dual_z += penalty * err_z
		dual_x1 += penalty * err_x1
		dual_x2 += penalty * err_x2

		# grad z, compute (1-gamma)H(Z)
		grad_z = -(1-gamma) * (np.log(pz)+1) - dual_z - penalty * err_z
		mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		# grad x1, compute -gamma H(Z|X_1)
		grad_x1 = -gamma * (np.log(pzcx1)+1) * px1[None,:] - dual_x1 - penalty * err_x1
		mean_grad_x1 = grad_x1 - np.mean(grad_x1,axis=0)
		grad_x2 = -gamma * (np.log(pzcx2)+1) * px2[None,:] - dual_x2 - penalty * err_x2
		mean_grad_x2 = grad_x2 - np.mean(grad_x2,axis=0)
		ss_z = gd.naiveStepSize(pz,-mean_grad_z, ss_init,ss_scale)
		if ss_z == 0:
			break
		ss_x1 = gd.naiveStepSize(pzcx1,-mean_grad_x1,ss_z,ss_scale)
		if ss_x1 == 0:
			break
		ss_x2 = gd.naiveStepSize(pzcx2,-mean_grad_x2,ss_x1,ss_scale)
		if ss_x2 == 0:
			break
		new_pz = pz - mean_grad_z * ss_x2
		new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x2
		new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2

		# error
		err_z = np.sum(pzcx1x2* px1x2[None,:,:],axis=(1,2)) - new_pz
		err_x1 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=2) - new_pzcx1
		err_x2 = np.sum(pzcx1x2 *(px2cx1.T)[None,:,:],axis=1) - new_pzcx2

		conv_z = 0.5 *np.sum(np.abs(err_z))
		conv_x1 = 0.5 *np.sum(np.abs(err_x1),axis=0)
		conv_x2 = 0.5 * np.sum(np.abs(err_x2),axis=0)
		if conv_z<convthres and np.all(conv_x1<convthres) and np.all(conv_x2<convthres):
			conv_flag = True
			break
		else:
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

def tfDetComAdmm(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs["ss_init"]
	ss_scale = kwargs["ss_scale"]
	penalty = kwargs['penalty_coeff']
	if kwargs["seed"]:
		rng = tf.random.Generator.from_seed(kwargs["seed"])
	else:
		rng = tf.random.Generator.from_non_deterministic_state()
	tf_px1x2 = tf.convert_to_tensor(px1x2,dtype=tf.float64)
	(nx1,nx2) = px1x2.shape
	px1 = tf.reduce_sum(tf_px1x2,axis=1)
	px2 = tf.reduce_sum(tf_px1x2,axis=0)
	px1cx2 = tf_px1x2/px2[None,:]
	px2cx1 = tf.transpose(tf_px1x2/px1[:,None])
	if "init_load" in kwargs.keys():
		pzcx1x2 = tf.convert_to_tensor(kwargs["pzcx1x2"],dtype=tf.float64)
		pz = tf.convert_to_tensor(kwargs["pz"],dtype=tf.float64)
		pzcx1 = tf.convert_to_tensor(kwargs["pzcx1"],dtype=tf.float64)
		pzcx2 = tf.convert_to_tensor(kwargs["pzcx2"],dtype=tf.float64)
		dual_z = tf.convert_to_tensor(kwargs["dual_z"],dtype=tf.float64)
		dual_x1 = tf.convert_to_tensor(kwargs["dual_x1"],dtype=tf.float64)
		dual_x2 = tf.convert_to_tensor(kwargs["dual_x2"],dtype=tf.float64)
	else:
		# random initialization
		pzcx1x2 = tf.nn.softmax(rng.normal(shape=(nz,nx1,nx2),dtype=tf.float64),axis=0)
		# augmented variables
		pz = tf.reduce_sum(pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2))
		pzcx1 = tf.reduce_sum(pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2)
		pzcx2 = tf.reduce_sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		# dual vars
		dual_z = tf.zeros(pz.shape,dtype=tf.float64)
		dual_x1= tf.zeros(pzcx1.shape,dtype=tf.float64)
		dual_x2= tf.zeros(pzcx2.shape,dtype=tf.float64)
	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt += 1
		err_z = tf.reduce_sum(pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = tf.reduce_sum(pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - pzcx1
		err_x2 = tf.reduce_sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# errors
		'''
		err_z = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		'''
		# gradient of convex part
		grad_p = (gamma * (tf.math.log(pzcx1x2)+1)\
				+ (dual_z + penalty * err_z)[:,None,None]\
				+ ((dual_x1 + penalty * err_x1)/px1[None,:])[:,:,None]\
				+ tf.repeat(tf.expand_dims((dual_x2+penalty*err_x2)/px2[None,:],1),repeats=nx1,axis=1) )*tf_px1x2[None,:,:]
		
		#grad_p = (gamma * (np.log(pzcx1x2)+1)\
		#		 + (dual_z+penalty*err_z)[:,None,None]\
		#		 + ((dual_x1+penalty*err_x1)/px1[None,:])[:,:,None]\
		#		 + np.repeat(np.expand_dims((dual_x2+penalty*err_x2)/px2[None,:],1),repeats=len(px1),axis=1) )*px1x2[None,:,:]
		#mean_grad_p = grad_p - np.mean(grad_p,axis=0)
		mean_grad_p = grad_p - tf.reduce_mean(grad_p,axis=0)
		#ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		ss_p = gd.tfNaiveSS(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		if ss_p == 0:
			break
		new_pzcx1x2 = pzcx1x2 - ss_p * mean_grad_p
		#if ss_p == 0:
		#	break
		#new_pzcx1x2 = pzcx1x2 - ss_p * mean_grad_p

		# new error
		#err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		#err_x1 = np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		#err_x2 = np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		err_z = tf.reduce_sum(new_pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = tf.reduce_sum(new_pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - pzcx1
		err_x2 = tf.reduce_sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# dual update
		#dual_z += penalty * err_z
		#dual_x1 += penalty * err_x1
		#dual_x2 += penalty * err_x2
		dual_z += penalty * err_z
		dual_x1 += penalty * err_x1
		dual_x2 += penalty * err_x2

		# grad z, compute (1-gamma)H(Z)
		#grad_z = -(1-gamma) * (np.log(pz)+1) - dual_z - penalty * err_z
		#mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		grad_z = -(1-gamma) * (tf.math.log(pz)+1) - dual_z - penalty * err_z
		mean_grad_z = grad_z - tf.reduce_mean(grad_z,axis=0)
		# grad x1, compute -gamma H(Z|X_1)
		#grad_x1 = -gamma * (np.log(pzcx1)+1) * px1[None,:] - dual_x1 - penalty * err_x1
		#mean_grad_x1 = grad_x1 - np.mean(grad_x1,axis=0)
		grad_x1 = -gamma * (tf.math.log(pzcx1)+1)* px1[None,:] - dual_x1 - penalty * err_x1
		mean_grad_x1 = grad_x1 - tf.reduce_mean(grad_x1,axis=0)
		#grad_x2 = -gamma * (np.log(pzcx2)+1) * px2[None,:] - dual_x2 - penalty * err_x2
		#mean_grad_x2 = grad_x2 - np.mean(grad_x2,axis=0)
		grad_x2 = -gamma * (tf.math.log(pzcx2)+1) * px2[None,:] - dual_x2 - penalty * err_x2
		mean_grad_x2 = grad_x2 - tf.reduce_mean(grad_x2,axis=0)
		#ss_z = gd.naiveStepSize(pz,-mean_grad_z, ss_init,ss_scale)
		ss_z = gd.tfNaiveSS(pz,-mean_grad_z,ss_init,ss_scale)
		if ss_z == 0:
			break
		#if ss_z == 0:
		#	break
		#ss_x1 = gd.naiveStepSize(pzcx1,-mean_grad_x1,ss_z,ss_scale)
		ss_x1 = gd.tfNaiveSS(pzcx1,-mean_grad_x1, ss_z, ss_scale)
		if ss_x1 == 0:
			break
		#if ss_x1 == 0:
		#	break
		#ss_x2 = gd.naiveStepSize(pzcx2,-mean_grad_x2,ss_x1,ss_scale)
		ss_x2 = gd.tfNaiveSS(pzcx2,-mean_grad_x2,ss_x1,ss_scale)
		if ss_x2 == 0:
			break
		#if ss_x2 == 0:
		#	break
		#new_pz = pz - mean_grad_z * ss_x2
		#new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x2
		#new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2
		new_pz = pz - mean_grad_z * ss_x2
		new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x2
		new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2

		# error
		#err_z = np.sum(pzcx1x2* px1x2[None,:,:],axis=(1,2)) - new_pz
		#err_x1 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=2) - new_pzcx1
		#err_x2 = np.sum(pzcx1x2 *(px2cx1.T)[None,:,:],axis=1) - new_pzcx2
		err_z = tf.reduce_sum(new_pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - new_pz
		err_x1 = tf.reduce_sum(new_pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - new_pzcx1
		err_x2 = tf.reduce_sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - new_pzcx2
		#conv_z = 0.5 *np.sum(np.abs(err_z))
		#conv_x1 = 0.5 *np.sum(np.abs(err_x1),axis=0)
		#conv_x2 = 0.5 * np.sum(np.abs(err_x2),axis=0)
		conv_z = 0.5 * tf.reduce_sum(tf.math.abs(err_z))
		conv_x1 = 0.5 * tf.reduce_sum(tf.math.abs(err_x1),axis=0)
		conv_x2 = 0.5 * tf.reduce_sum(tf.math.abs(err_x2),axis=0)
		if tf.reduce_all(conv_z < convthres) and tf.reduce_all(conv_x1 < convthres) and tf.reduce_all(conv_x2 < convthres):
			conv_flag = True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			pz = new_pz
			pzcx1 = new_pzcx1
			pzcx2 = new_pzcx2

	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2.numpy(),
		"pz":pz.numpy(),"pzcx1":pzcx1.numpy(),"pzcx2":pzcx2.numpy(),
		"dual_z":dual_z.numpy(),"dual_x1":dual_x1.numpy(),"dual_x2":dual_x2.numpy(),}

def tfStoComAdmm(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale= kwargs['ss_scale']
	penalty = kwargs['penalty_coeff']
	#d_seed = None
	if kwargs["seed"]:
		rng = tf.random.Generator.from_seed(kwargs["seed"])
	else:
		rng = tf.random.Generator.from_non_deterministic_state()
	tf_px1x2 = tf.convert_to_tensor(px1x2,dtype=tf.float64)
	(nx1,nx2) = px1x2.shape
	#px2 = np.sum(px1x2,0)
	#px1 = np.sum(px1x2,1)
	px1 = tf.reduce_sum(tf_px1x2,axis=1)
	px2 = tf.reduce_sum(tf_px1x2,axis=0)
	px1cx2 = tf_px1x2 /px2[None,:]
	px2cx1 = tf.transpose(tf_px1x2/ px1[:,None])

	if "init_load" in kwargs.keys():
		pzcx1x2 = tf.convert_to_tensor(kwargs["pzcx1x2"],dtype=tf.float64)
		pz = tf.convert_to_tensor(kwargs["pz"],dtype=tf.float64)
		pzcx1 = tf.convert_to_tensor(kwargs["pzcx1"],dtype=tf.float64)
		pzcx2 = tf.convert_to_tensor(kwargs["pzcx2"],dtype=tf.float64)
		dual_z = tf.convert_to_tensor(kwargs["dual_z"],dtype=tf.float64)
		dual_x1 = tf.convert_to_tensor(kwargs["dual_x1"],dtype=tf.float64)
		dual_x2 = tf.convert_to_tensor(kwargs["dual_x2"],dtype=tf.float64)
	else:
		# random initialization
		pzcx1x2 = tf.nn.softmax(rng.normal(shape=(nz,nx1,nx2),dtype=tf.float64),axis=0)
		#pzcx1x2 = rng.random((nz,nx1,nx2))
		#pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# augmented variables
		pz = tf.reduce_sum(pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2))
		pzcx1 = tf.reduce_sum(pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2)
		pzcx2 = tf.reduce_sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
		#pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		#pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		#pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		# dual vars
		dual_z = tf.zeros(pz.shape,dtype=tf.float64)
		dual_x1= tf.zeros(pzcx1.shape,dtype=tf.float64)
		dual_x2= tf.zeros(pzcx2.shape,dtype=tf.float64)
	#print(pzcx1x2)
	#print(pzcx1.shape)
	#print(pzcx2.shape)
	#sys.exit()
	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt += 1
		#err_z = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_z = tf.reduce_sum(pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - pz
		#err_x1= np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x1 = tf.reduce_sum(pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - pzcx1
		#err_x2= np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		err_x2 = tf.reduce_sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# compute the gradient 

		#grad_p = ( (1+gamma) * (np.log(pzcx1x2)+1) \
		#		+ (dual_z + penalty * err_z)[:,None,None]\
		#		+ ((dual_x1 + penalty * err_x1)/px1[None,:])[:,:,None]\
		#		+np.repeat(np.expand_dims((dual_x2 + penalty * err_x2)/px2[None,:],axis=1),repeats=len(px1),axis=1) )*px1x2[None,:,:]
		grad_p = ( (1+gamma) * (tf.math.log(pzcx1x2)+1)\
				+ (dual_z + penalty * err_z)[:,None,None]\
				+ ((dual_x1 + penalty * err_x1)/px1[None,:])[:,:,None]\
				+ tf.repeat(tf.expand_dims((dual_x2 + penalty * err_x2)/px2[None,:],axis=1),repeats=len(px1),axis=1) )*tf_px1x2[None,:,:]

		#mean_grad_p = grad_p - np.mean(grad_p,axis=0)
		mean_grad_p = grad_p - tf.reduce_mean(grad_p,axis=0)
		#ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		ss_p = gd.tfNaiveSS(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		if ss_p == 0:
			break
		new_pzcx1x2 = pzcx1x2 - mean_grad_p * ss_p
		# new errors
		#err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		#err_x1= np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		#err_x2= np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		err_z = tf.reduce_sum(new_pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = tf.reduce_sum(new_pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - pzcx1
		err_x2 = tf.reduce_sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# daul updates
		dual_z += penalty * err_z
		dual_x1+= penalty * err_x1
		dual_x2+= penalty * err_x2

		# compute the augmented gradients
		#grad_z = -(1-gamma) * (np.log(pz)+1) - dual_z - penalty*err_z
		#mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		#ss_z = gd.naiveStepSize(pz,-mean_grad_z,ss_init,ss_scale)
		grad_z = -(1-gamma) * (tf.math.log(pz)+1) - dual_z - penalty * err_z
		mean_grad_z = grad_z - tf.reduce_mean(grad_z,axis=0)
		grad_x1 = -gamma * (tf.math.log(pzcx1)+1)*px1[None,:] - dual_x1 - penalty * err_x1
		mean_grad_x1 = grad_x1 - tf.reduce_mean(grad_x1,axis=0)
		grad_x2 = -gamma * (tf.math.log(pzcx2)+1)*px2[None,:] - dual_x2 - penalty * err_x2
		mean_grad_x2 = grad_x2 - tf.reduce_mean(grad_x2,axis=0)
		ss_z = gd.tfNaiveSS(pz,-mean_grad_z,ss_init,ss_scale)
		if ss_z == 0:
			break
		#grad_x1= -gamma  * (np.log(pzcx1)+1) - dual_x1 - penalty * err_x1
		#mean_grad_x1 = grad_x1 - np.mean(grad_x1,axis=0)
		#ss_x1 = gd.naiveStepSize(pzcx1,-mean_grad_x1,ss_z,ss_scale)
		ss_x1 = gd.tfNaiveSS(pzcx1,-mean_grad_x1,ss_z,ss_scale)
		if ss_x1 == 0:
			break
		#grad_x2 = -gamma * (np.log(pzcx2)+1) - dual_x2 - penalty * err_x2
		#mean_grad_x2 = grad_x2 - np.mean(grad_x2,axis=0)
		#ss_x2 = gd.naiveStepSize(pzcx2,-mean_grad_x2,ss_x1,ss_scale)
		ss_x2 = gd.tfNaiveSS(pzcx2,-mean_grad_x2,ss_x1,ss_scale)
		if ss_x2 == 0:
			break
		new_pz = pz - mean_grad_z * ss_x2
		new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x2
		new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2
		# new errors
		#err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - new_pz
		#err_x1= np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - new_pzcx1
		#err_x2= np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - new_pzcx2
		err_z = tf.reduce_sum(new_pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - new_pz
		err_x1 = tf.reduce_sum(new_pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - new_pzcx1
		err_x2 = tf.reduce_sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - new_pzcx2
		# convergence
		#conv_z = np.sum(np.abs(err_z))
		#conv_x1 = np.sum(np.abs(err_x1),axis=0)
		#conv_x2 = np.sum(np.abs(err_x2),axis=0)
		conv_z = 0.5*tf.reduce_sum(tf.math.abs(err_z))
		conv_x1 = 0.5*tf.reduce_sum(tf.math.abs(err_x1),axis=0)
		conv_x2 = 0.5*tf.reduce_sum(tf.math.abs(err_x2),axis=0)
		if tf.reduce_all(conv_z< convthres) and tf.reduce_all(conv_x1<convthres) and tf.reduce_all(conv_x2<convthres):
			conv_flag = True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			pz = new_pz
			pzcx1 = new_pzcx1
			pzcx2 = new_pzcx2

	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2.numpy(),
			"pz":pz.numpy(),"pzcx1":pzcx1.numpy(),"pzcx2":pzcx2.numpy(),
			"dual_z":dual_z.numpy(),"dual_x1":dual_x1.numpy(),"dual_x2":dual_x2.numpy()}


