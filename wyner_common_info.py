import numpy as np
import os
import sys
import pickle
import utils as ut
import algorithm as alg
import dataset as dt
import datetime
import copy
import argparse
import evaluation as ev
import hessian as hs


parser = argparse.ArgumentParser()
parser.add_argument("method",choices=["wyner",'wynerdca'])
parser.add_argument("--maxiter",type=int,default=100000,help="maximum iteration before termination")
parser.add_argument("--convthres",type=float,default=1e-6,help="convergence threshold")
parser.add_argument("--nrun",type=int,default=10,help="number of trail of each simulation")
parser.add_argument("--ss_init",type=float,default=1e-1,help="step size initialization")
parser.add_argument("--ss_scale",type=float,default=0.25,help="step size scaling")
parser.add_argument("--output_dir",type=str,default="wyner_com_results",help="output filename")
parser.add_argument("--seed",type=int,default=None,help="random seed for reproduction")
parser.add_argument("--ny",type=int,default=2,help="number of uniform hidden labels")
parser.add_argument("--nb",type=int,default=2,help="number of blocks for observations")
parser.add_argument("--corr",type=float,default=0,help="cyclic observation uncertainty given a label")
#parser.add_argument("--startz",type=int,default=2,help="starting number of nz search")
#parser.add_argument("--endz",type=int,default=4,help="ending number of nz search")
#parser.add_argument("--stepz",type=int,default=1,help="stepsize of nz search")
#parser.add_argument('--beta',type=float,default=8.0,help="KL divergence trade-off parameter")
parser.add_argument("--betamin",type=float,default=0.1,help="minimum trade-off parameter for search")
parser.add_argument("--betamax",type=float,default=2.5,help="maximum trade-off parameter for search")
parser.add_argument("--numbeta",type=int,default=20,help="number of search points")


args = parser.parse_args()
argsdict = vars(args)

print(argsdict)

data_dict = dt.synExpandToy(args.ny,args.nb,args.corr)
prob_joint = data_dict["p_joint"]
px1 = np.sum(prob_joint,1)
px2 = np.sum(prob_joint,0)
mix1x2 = ut.calcMI(prob_joint)
print("mix1x2={:.4f}".format(mix1x2))
print("Hx1={:.4f}".format(ut.calcEnt(px1)))
print("Hx2={:.4f}".format(ut.calcEnt(px2)))
print("Hx1x2={:.4f}".format(ut.calcEnt(prob_joint)))
prob_cond = data_dict["p_cond"]
px1cx2 = prob_cond[0]
px2cx1 = prob_cond[1]

gamma_range = np.geomspace(args.betamin,args.betamax,num=args.numbeta)

alg_dict = {
"ss_init":args.ss_init,
"ss_scale":args.ss_scale,
"seed":args.seed,
}

if args.method=="wyner":
	algrun = alg.wynerDrs
elif args.method == "wynerdca":
	algrun = alg.wynerDCA
else:
	sys.exit("undefined method {:}".format(args.method))

nz_set = np.array([args.ny]) # FIXME: assume knowing the cardinality of 
#nz_set = np.arange(max(2,args.startz),args.endz+1,args.stepz)
for beta in gamma_range:
	for nz in nz_set:
		for nn in range(args.nrun):
			out_dict = algrun(prob_joint,nz,beta,args.maxiter,args.convthres,**alg_dict)
			# calculate the mutual informations
			pz = out_dict["pz"]
			entz = ut.calcEnt(pz)
			pzcx1x2 = out_dict['pzcx1x2'] # this might not be a valid prob, but is forced to be valid one
			est_pzx1x2 = out_dict['est_pzx1x2']
			pzx1x2 = est_pzx1x2
			# KL distance
			est_px1x2 = np.sum(pzx1x2,axis=0)
			dkl_error = ut.calcKL(prob_joint,est_px1x2)

			entzcx1x2 = np.sum(-pzx1x2 * np.log(pzcx1x2))
			joint_mi = entz - entzcx1x2
			# take the maximum element
			mizx1 = ut.calcMI(np.sum(pzx1x2,axis=2))
			mizx2 = ut.calcMI(np.sum(pzx1x2,axis=1))
			#cmix1x2cz = ut.calcMIcond(np.transpose(pzx1x2,(1,2,0))) # this is faulty
			# by definition 
			cmix1x2cz = joint_mi - (mizx1+mizx2) + mix1x2

			# loss calculation
			
			tmp_loss = joint_mi - beta * (mizx1 + mizx2) # for recording
			eigvals = hs.computeHessian_2views(pzcx1x2,prob_joint)
			eig_v2 = eigvals[:,-2]
			eig_tex = ",".join(["{:.4e}".format(item) for item in eig_v2])
			print("nz,{:},beta,{:>6.3f},conv,{:},niter,{:>4},I12,{:.4e},I1,{:.4e},I2,{:.4e},IX1X2cZ,{:.5e},Dkl_x12,{:.4e},{:}".format(
					nz,
					beta,
					out_dict['conv'],
					out_dict['niter'],
					joint_mi,
					mizx1,
					mizx2,
					cmix1x2cz,
					dkl_error,
					eig_tex
				))

