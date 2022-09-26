import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import algorithm as alg
import utils as ut

# data
px1cx2 = np.array([
	[0.85, 0.04, 0.06],
	[0.07, 0.88, 0.02],
	[0.08, 0.08, 0.92]])

px2 = np.array([0.25, 0.35,0.40])
px12 = px1cx2 * px2[None,:]
px1 = np.sum(px12,1)
px2cx1 = (px12/px1[:,None]).T

# variables
nz = 2
nx1 = len(px1)
nx2 = len(px2)

rng = np.random.default_rng()
pzcx1x2 = rng.random((nz,nx1,nx2))
pzcx1x2 = pzcx1x2/ np.sum(pzcx1x2,0)
#print(pzcx1x2)

# precompute the priors
pz = np.sum(pzcx1x2 * px12[None,...],axis=(1,2))
pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,...],axis=2)
pzcx2 = np.sum(pzcx1x2 * px1cx2[None,...],axis=1)
#print("sum pz:",np.sum(pz))
#print("sum row pzcx1",np.sum(pzcx1,axis=0))
#print("sum row pzcx2",np.sum(pzcx2,axis=0))

pzx1x2 = pzcx1x2 * px12[None,...]

# we can precompute the metrics...
# let's check
mizx1 = ut.calcMI(pzcx1 * px1[None,:])
mizx2 = ut.calcMI(pzcx2 * px2[None,:])

# always compute components
# MI(X1X2;Z) = MI(X1;Z) + MI(X2;Z|X1)
cmizx2cx1 = ut.calcMIcond(np.transpose(pzx1x2,(0,2,1)))
# MI(X1X2;Z) = MI(X2;Z) + MI(X1;Z|X2)
cmizx1cx2 = ut.calcMIcond(pzx1x2)
print("mizx1 + cmizx2cx1=",mizx1 + cmizx2cx1)
print("mizx2 + cmizx1cx2=",mizx2 + cmizx1cx2)

gamma1 = 0.1000
gamma2 = 0.1500
maxiter = 50000
convthres = 1e-6
conv_flag = False

# try admm method
ss_init = 4e-3
ss_scale = 0.25
penalty_coeff = 64.0
admm_param = {"ss_init":ss_init,"ss_scale":ss_scale,"penalty_coeff":penalty_coeff}
#out_dict = alg.admmsolver(px12,nz,gamma,maxiter,convthres,**admm_param)
out_dict = alg.distAdmmSolver(px12,nz,gamma1,gamma2,maxiter,convthres,**admm_param)
print("convergence:",out_dict["conv"])
print("niter",out_dict["niter"])
print("pzcx1")
print(out_dict['pzcx1'])
print("pzcx2")
print(out_dict["pzcx2"])