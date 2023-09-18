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
from scipy.io import savemat


parser = argparse.ArgumentParser()
# TODO: should include other solvers and their hyper parameters
# For example: step-size of gradients, penalty coefficients
parser.add_argument("method",choices=["wyner",'wynerdca',"logdrsvar"])
parser.add_argument("--maxiter",type=int,default=50000,help="maximum iteration before termination")
parser.add_argument("--convthres",type=float,default=1e-6,help="convergence threshold")
parser.add_argument("--nrun",type=int,default=100,help="number of trail of each simulation")
parser.add_argument("--ss_init",type=float,default=1e-1,help="step size initialization")
parser.add_argument("--ss_scale",type=float,default=0.25,help="step size scaling")
parser.add_argument("--penalty",type=float,default=128.0,help="penalty coefficient for ADMM-based solvers")
parser.add_argument("--seed",type=int,default=None,help="random seed for reproduction")
parser.add_argument("--ny",type=int,default=2,help="number of uniform hidden labels")
parser.add_argument("--nb",type=int,default=2,help="number of blocks for observations")
parser.add_argument("--corr",type=float,default=0,help="cyclic observation uncertainty given a label")

parser.add_argument("--betamin",type=float,default=0.1,help="minimum trade-off parameter for search")
parser.add_argument("--betamax",type=float,default=10.0,help="maximum trade-off parameter for search")
parser.add_argument("--numbeta",type=int,default=40,help="number of search points")

parser.add_argument("--savedir",type=str,default="exp_global_test",help="save directory")
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

# get the global solution 
global_pycx1x2 = ut.computeGlobalSolution(
	data_dict['pxcy_list'][0],
	data_dict['pxcy_list'][1],
	data_dict['py'])
#print(global_pycx1x2)
#sys.exit()

gamma_range = np.geomspace(args.betamin,args.betamax,num=args.numbeta)

alg_dict = {
"ss_init":args.ss_init,
"ss_scale":args.ss_scale,
"seed":args.seed,
"penalty_coeff":args.penalty,
}

if args.method=="wyner":
	algrun = alg.wynerDrs
elif args.method == "wynerdca":
	algrun = alg.wynerDCA
elif args.method == "logdrsvar":
	algrun = alg.stoLogDrsVar
else:
	sys.exit("undefined method {:}".format(args.method))


nz_set = np.array([args.ny]) # FIXME: assume knowing the cardinality of 

hdr_tex = "nz_beta_conv_sgblDtv_niter_I12_I1_I2_IC_KLerr"
record_arr = np.zeros((len(gamma_range)*len(nz_set)*args.nrun,10))

#nz_set = np.arange(max(2,args.startz),args.endz+1,args.stepz)
ridx =0
for beta in gamma_range:
	for nz in nz_set:
		for nn in range(args.nrun):
			out_dict = algrun(prob_joint,nz,beta,args.maxiter,args.convthres,**alg_dict)
			# calculate the mutual informations
			pz = out_dict["pz"]
			entz = ut.calcEnt(pz)
			pzcx1x2 = out_dict['pzcx1x2'] # this might not be a valid prob, but is forced to be valid one

			# total variation distance to the global solution
			# NOTE: there is a label matching problem that needs to be solved
			corrected_pzcx1x2,global_dtv = ev.compare_global_solution(global_pycx1x2,pzcx1x2)
			sum_global_dtv = np.sum(prob_joint*global_dtv) # expected DTV

			est_pzx1x2 = out_dict['est_pzx1x2']
			pzx1x2 = est_pzx1x2
			# KL distance
			est_px1x2 = np.sum(pzx1x2,axis=0)
			dkl_error = ut.calcKL(prob_joint,est_px1x2)

			entzcx1x2 = np.sum(-pzx1x2 * np.log(pzcx1x2))
			# take the maximum element
			mizx1 = ut.calcMI(np.sum(pzx1x2,axis=2))
			mizx2 = ut.calcMI(np.sum(pzx1x2,axis=1))
			cmix1x2cz = ut.calcMIcond(np.transpose(pzx1x2,(1,2,0)))
			if cmix1x2cz > 1e1:
				print("unstable")
				print(pzcx1x2)
				sys.exit("debugging")
			# loss calculation
			joint_mi = entz - entzcx1x2
			tmp_loss = joint_mi - beta * (mizx1 + mizx2) # for recording
			# SECOND ORDER ANALYSIS
			#eigvals = hs.computeHessian_2views(pzcx1x2,prob_joint)
			#eig_v2 = eigvals[:,-2]
			#eig_tex = ",".join(["{:.4e}".format(item) for item in eig_v2])
			# COMPUTE the distance to the global solution

			print("beta,{:>6.3f},conv,{:},global,{:.4e},niter,{:>4},I12,{:.4e},IX1X2cZ,{:.5e},Dkl_x12,{:.4e}".format(
					beta,
					out_dict['conv'],
					sum_global_dtv,
					out_dict['niter'],
					joint_mi,
					cmix1x2cz,
					dkl_error,
					#eig_tex
				))
			# record the results
			record_arr[ridx,:] = np.array([
				nz,
				beta,
				int(out_dict['conv']),
				sum_global_dtv,
				out_dict['niter'],
				joint_mi,
				mizx1,
				mizx2,
				cmix1x2cz,
				dkl_error,])
			ridx+=1


d_save_dir = os.path.join(os.getcwd(),args.savedir)
os.makedirs(d_save_dir,exist_ok=True)

tnow = datetime.datetime.now()
repeat_cnt = 0
safe_savename_base = "globalTest_{:}_y{:}b{:}_cr{:.4f}_{:}".format(args.method,args.ny,args.nb,args.corr,tnow.strftime("%Y%m%d"))
safe_savename = copy.copy(safe_savename_base)
while os.path.isfile(os.path.join(d_save_dir,safe_savename)+".mat"):
	repeat_cnt+=1
	safe_savename = "{:}_{:}".format(safe_savename_base,repeat_cnt)

with open(os.path.join(d_save_dir,safe_savename+"_config.pkl"),'wb') as fid:
	pickle.dump(argsdict,fid)
with open(os.path.join(d_save_dir,safe_savename+".npy"),'wb') as fid:
	np.save(fid,record_arr)
savemat(os.path.join(d_save_dir,safe_savename+".mat"),{"label":hdr_tex,"result_array":record_arr})

print("Save the results to:{:}".format(os.path.join(d_save_dir,safe_savename+".mat")))
