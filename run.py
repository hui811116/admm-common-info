import numpy as np
import os
import sys
import argparse
import pickle
import utils as ut
import algorithm as alg
import dataset as dt

parser = argparse.ArgumentParser()
parser.add_argument("--penalty",type=float,default=128.0,help="penalty coefficient of the ADMM solver")
parser.add_argument("--maxiter",type=int,default=50000,help="maximum iteration before termination")
parser.add_argument("--convthres",type=float,default=1e-6,help="convergence threshold")
parser.add_argument("--nrun",type=int,default=10,help="number of trail of each simulation")
parser.add_argument("--ss_init",type=float,default=1e-3,help="step size initialization")
parser.add_argument("--ss_scale",type=float,default=0.25,help="step size scaling")
parser.add_argument("--output",type=str,default="unsu_mv_output",help="output filename")
parser.add_argument("--seed",type=int,default=None,help="random seed for reproduction")
parser.add_argument("--dataset",type=str,default="toy2v",help="specify the dataset to use")
parser.add_argument("--gamma_min",type=float,default=0.1,help="minimum gamma value")
parser.add_argument("--gamma_max",type=float,default=1.0,help="maximum gamma value")
# the maximum value is always 1. otherwise a different problem
parser.add_argument("--gamma_num",type=int,default=10,help="number of gamma values")

args = parser.parse_args()
argsdict = vars(args)

print(argsdict)

data_dict = dt.getDataset(args.dataset)
prob_joint = data_dict["p_joint"]
px1 = np.sum(prob_joint,1)
px2 = np.sum(prob_joint,0)
mix1x2 = ut.calcMI(prob_joint)
print("mix1x2={:.4f}".format(mix1x2))
print("Hx1={:.4f}".format(ut.calcEnt(px1)))
print("Hx2={:.4f}".format(ut.calcEnt(px2)))
print("Hx1x2={:.4f}".format(ut.calcEnt(prob_joint)))
#sys.exit()
prob_cond = data_dict["p_cond"]
px1cx2 = prob_cond[0]
px2cx1 = prob_cond[1]
nz_list = prob_joint.shape

#gamma_range = np.geomspace(1,args.gamma_max,num=args.gamma_num)
gamma_range = np.geomspace(args.gamma_min,args.gamma_max,num=args.gamma_num)

alg_dict = {
"penalty_coeff":args.penalty,
"ss_init":args.ss_init,
"ss_scale":args.ss_scale,
"seed":args.seed,
}

nz_set = np.arange(2,np.prod(nz_list)+1)
res_all = np.zeros((len(gamma_range)*args.nrun*len(nz_set),10)) # gamma, nidx, niter, conv,nz, entz, mizx1,mizx2,cmizx1cx2, cmizx2cx1
rec_idx = 0
for gidx ,gamma in enumerate(gamma_range):
	for nz in nz_set:
		conv_cnt = 0
		for nn in range(args.nrun):			
			out_dict = alg.detComAdmm(prob_joint,nz,gamma,args.maxiter,args.convthres,**alg_dict)
			tmp_result = [gamma,nn,out_dict['niter'],int(out_dict["conv"]),nz]
			conv_cnt += int(out_dict['conv'])
			if out_dict['conv']:
				# convergence reached... do things afterward
				# calculate the mutual informations
				pzcx1x2 = out_dict["pzcx1x2"]
				pz = np.sum(pzcx1x2 * prob_joint[None,:,:],axis=(1,2))
				entz = ut.calcEnt(pz)
				pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
				pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
				mizx1 = ut.calcMI(pzcx1 * px1[None,:])
				mizx2 = ut.calcMI(pzcx2 * px2[None,:])
				cmizx1cx2 = ut.calcMIcond(pzcx1x2 * prob_joint[None,:,:])
				# calculate other direction
				cmizx2cx1 = ut.calcMIcond(np.transpose(pzcx1x2 * prob_joint[None,:,:],(0,2,1)))
				tmp_result += [entz,mizx1,mizx2,cmizx1cx2,cmizx2cx1]
			else:
				# failed in convergence, either maximum iteration reached or no available step size
				tmp_result += [0,0,0,0,0]
			res_all[rec_idx,:] = np.array(tmp_result)
			rec_idx += 1
		print("simulated gamma:{:.4f}, nz:{:}, conv_rate:{:.4f}".format(gamma,nz,conv_cnt/args.nrun))


# saving the results, numpy array
with open(args.output+".npy","wb") as fid:
	np.save(fid,res_all)