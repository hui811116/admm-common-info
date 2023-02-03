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


parser = argparse.ArgumentParser()
parser.add_argument("method",choices=["wyner",'wlogdrs'])
parser.add_argument("--penalty",type=float,default=128.0,help="penalty coefficient of the ADMM solver")
parser.add_argument("--maxiter",type=int,default=50000,help="maximum iteration before termination")
parser.add_argument("--convthres",type=float,default=1e-6,help="convergence threshold")
parser.add_argument("--nrun",type=int,default=25,help="number of trail of each simulation")
parser.add_argument("--ss_init",type=float,default=1e-1,help="step size initialization")
parser.add_argument("--ss_scale",type=float,default=0.25,help="step size scaling")
parser.add_argument("--output_dir",type=str,default="wyner_sto_results",help="output filename")
parser.add_argument("--seed",type=int,default=None,help="random seed for reproduction")
parser.add_argument("--ny",type=int,default=2,help="number of uniform hidden labels")
parser.add_argument("--nb",type=int,default=2,help="number of blocks for observations")
parser.add_argument("--corr",type=float,default=0,help="cyclic observation uncertainty given a label")
parser.add_argument("--startz",type=int,default=2,help="starting number of nz search")
parser.add_argument("--endz",type=int,default=4,help="ending number of nz search")
parser.add_argument("--stepz",type=int,default=1,help="stepsize of nz search")
parser.add_argument('--beta',type=float,default=8.0,help="KL divergence trade-off parameter")

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

gamma_range = np.array([1])

alg_dict = {
"penalty_coeff":args.penalty,
"ss_init":args.ss_init,
"ss_scale":args.ss_scale,
"seed":args.seed,
}

if args.method=="wyner":
	algrun = alg.wynerDrs
elif args.method == "wlogdrs":
	algrun = alg.wynerDrsTrue
else:
	sys.exit("undefined method {:}".format(args.method))

#nz_set = np.array([args.ny]) # FIXME: assume knowing the cardinality of 
nz_set = np.arange(max(2,args.startz),args.endz+1,args.stepz)
res_all = np.zeros((args.nrun*len(nz_set),10)) # nidx, niter, conv,nz, entz, mizx1,mizx2,joint_MI, dkl_error, cmix1x2cz
rec_idx = 0
for nz in nz_set:
	for nn in range(args.nrun):
		out_dict = algrun(prob_joint,nz,args.beta,args.maxiter,args.convthres,**alg_dict)
		tmp_result = [nn,out_dict['niter'],int(out_dict["conv"]),nz]
		# convergence reached... do things afterward
		# calculate the mutual informations
		pz = out_dict["pz"]
		entz = ut.calcEnt(pz)
		# calculate ent_z|x1x2
		pzcx1 = out_dict["pzcx1"]
		pzcx2 = out_dict["pzcx2"]
		# compute the joint encoder
		pzcx1x2 = out_dict['pzcx1x2'] # this might not be a valid prob, but is forced to be valid one
		# only for wyner method
		est_pzx1x2 = out_dict['est_pzx1x2']
		pzx1x2 = est_pzx1x2
		entzcx1x2 = np.sum(-pzx1x2 * np.log(pzcx1x2))
		# take the maximum element
		mizx1 = ut.calcMI(np.sum(pzx1x2,axis=2))
		mizx2 = ut.calcMI(np.sum(pzx1x2,axis=1))
		cmix1x2cz = ut.calcMIcond(np.transpose(pzx1x2,(1,2,0)))
		# loss calculation
		joint_mi = entz - entzcx1x2
		dkl_error = np.sum(prob_joint * (np.log(prob_joint)-np.log(np.sum(est_pzx1x2,axis=0))))
		tmp_result += [entz,mizx1,mizx2,joint_mi,dkl_error,cmix1x2cz]
			
		res_all[rec_idx,:] = np.array(tmp_result)
		print("nidx,{:},nz,{:},conv,{:},nit,{:},IX12_Z,{:.4f},HZ,{:.4f},Error,{:.5f},IX1_X2|Z,{:.5f}".format(
			nn,nz,int(out_dict["conv"]),out_dict["niter"],
			joint_mi,entz,dkl_error,cmix1x2cz))
		rec_idx += 1

timenow= datetime.datetime.now()
# result directory
d_cwd = os.getcwd()
d_save_dir = os.path.join(d_cwd,args.output_dir)
os.makedirs(d_save_dir,exist_ok=True)

repeat_cnt = 0
safe_savename_base = "sto_wyner_y{:}b{:}_cr{:.4f}_c{:}_si{:.4e}_nzall_{:}".format(args.ny,args.nb,args.corr,args.penalty,args.ss_init,timenow.strftime("%Y%m%d"))
safe_savename = copy.copy(safe_savename_base)
while os.path.isfile(os.path.join(d_save_dir,safe_savename+".npy")):
	repeat_cnt +=1
	safe_savename = "{:}_{:}".format(safe_savename_base,repeat_cnt)

# saving the results, numpy array
with open(os.path.join(d_save_dir,safe_savename+".npy"),"wb") as fid:
	np.save(fid,res_all)
# saving the configuration in case of error
with open(os.path.join(d_save_dir,safe_savename+"_config.pkl"),"wb") as fid:
	pickle.dump(argsdict,fid)