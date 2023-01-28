import numpy as np
import os
import sys
import argparse
import pickle
import utils as ut
import algorithm as alg
import dataset as dt
import evaluation as ev
import datetime
import copy

parser = argparse.ArgumentParser()
parser.add_argument("method",choices=["stogd",'stologdrs','detlogdrs'])
parser.add_argument("dataset",type=str,help="dataset generated by script sample_gen.py")
parser.add_argument("--penalty",type=float,default=128.0,help="penalty coefficient of the ADMM solver")
parser.add_argument("--maxiter",type=int,default=200000,help="maximum iteration before termination")
parser.add_argument("--convthres",type=float,default=1e-6,help="convergence threshold")
parser.add_argument("--nrun",type=int,default=5,help="number of trail of each simulation")
parser.add_argument("--ss_init",type=float,default=1e-2,help="step size initialization")
parser.add_argument("--ss_scale",type=float,default=0.25,help="step size scaling")
parser.add_argument("--output_dir",type=str,default="unsu_test_output",help="output filename")
parser.add_argument("--seed",type=int,default=None,help="random seed for reproduction")
parser.add_argument("--gamma_min",type=float,default=1.0,help="minimum gamma value")
parser.add_argument("--gamma_max",type=float,default=50.0,help="maximum gamma value")
# the maximum value is always 1. otherwise a different problem
parser.add_argument("--gamma_num",type=int,default=20,help="number of gamma values")

args = parser.parse_args()
argsdict = vars(args)

print(argsdict)

dataset_dict = dt.loadSampleData(args.dataset)
dataname  = ".".join((os.path.split(args.dataset)[1]).split(".")[:-1])
print(dataname)
y_train, x_train = dataset_dict['ytrain'],dataset_dict['xtrain']
y_test, x_test   = dataset_dict['ytest'], dataset_dict['xtest']

# smoothing for stability
prob_joint = dataset_dict["p_joint"]
smooth_eps = 1e-9
prob_joint += smooth_eps
prob_joint/= np.sum(prob_joint)

px1 = np.sum(prob_joint,1)
px2 = np.sum(prob_joint,0)
mix1x2 = ut.calcMI(prob_joint)
print("mix1x2={:.4f}".format(mix1x2))
print("Hx1={:.4f}".format(ut.calcEnt(px1)))
print("Hx2={:.4f}".format(ut.calcEnt(px2)))
print("Hx1x2={:.4f}".format(ut.calcEnt(prob_joint)))

px1cx2 = prob_joint/px2[None,:]
px2cx1 = (prob_joint/px1[:,None]).T
# TODO: what if |Z| neq |Y|
nz = dataset_dict['ny']

gamma_range = np.geomspace(args.gamma_min,args.gamma_max,num=args.gamma_num)

alg_dict = {
"penalty_coeff":args.penalty,
"ss_init":args.ss_init,
"ss_scale":args.ss_scale,
"seed":args.seed,
}

# algorithm selection
if args.method == "stogd":
	algrun = alg.stoGradComp
elif args.method == "stologdrs":
	algrun = alg.stoLogDrsVar
elif args.method == 'detlogdrs':
	algrun = alg.detLogDrs
else:
	sys.exit("undefined method {:}".format(args.method))
nz_set = [nz]
# gamma, nidx, niter, conv,nz, entz, mizx1,mizx2,joint_MI, cmix1x2cz, train_acc, test_acc
res_all = np.zeros((len(gamma_range)*args.nrun*len(nz_set),12)) 
rec_idx = 0
for gidx ,gamma in enumerate(gamma_range):
	for nz in nz_set:
		for nn in range(args.nrun):
			out_dict = algrun(prob_joint,nz,gamma,args.maxiter,args.convthres,**alg_dict)
			tmp_result = [gamma,nn,out_dict['niter'],int(out_dict["conv"]),nz]
				
			# convergence reached... do things afterward
			# compute the joint encoder
			pzcx1x2 = out_dict['pzcx1x2'] # this might not be a valid prob, but is forced to be valid one
			pzx1x2 = pzcx1x2 * prob_joint[None,:,:]
			pz = np.sum(pzx1x2,axis=(1,2))
			entz = np.sum(-pz*np.log(pz))
			entzcx1x2 = np.sum(-pzx1x2*np.log(pzcx1x2))
			joint_mi = entz- entzcx1x2

			cmix1x2z = ut.calcMIcond(np.transpose(pzx1x2,axes=[1,2,0]))
			# take the maximum element
			pzcx1 = np.sum(pzx1x2,axis=2)/px1[None,:]
			pzcx2 = np.sum(pzx1x2,axis=1)/px2[None,:]

			mizx1 = ut.calcMI(pzcx1 * px1[None,:])
			mizx2 = ut.calcMI(pzcx2 * px2[None,:])
			cmix1x2cz = ut.calcMIcond(np.transpose(pzcx1x2 * prob_joint[None,:,:],(1,2,0)))
			# evaluation
			train_acc, test_acc = ev.evalSamples2V(y_train,x_train,y_test,x_test,pzcx1x2)
			# collecting results
			tmp_result += [entz,mizx1,mizx2,joint_mi,cmix1x2cz,train_acc,test_acc]
				
			res_all[rec_idx,:] = np.array(tmp_result)
			print("ga,{:.3f},nidx,{:},nz,{:},conv,{:},nit,{:},IX12_Z,{:.4f},HZ,{:.4f},IX1_2|Z,{:.5f},tr_acc,{:.4f},ts_acc,{:.4f}".format(
				gamma,nn,nz,int(out_dict["conv"]),out_dict["niter"],
				joint_mi,entz,cmix1x2cz,train_acc,test_acc))
			rec_idx += 1

timenow= datetime.datetime.now()
# result directory
d_cwd = os.getcwd()
d_save_dir = os.path.join(d_cwd,args.output_dir)
os.makedirs(d_save_dir,exist_ok=True)

repeat_cnt = 0
safe_savename_base = "unsutest_{:}_dt_{:}_{:}".format(args.method,dataname,timenow.strftime("%Y%m%d"))
safe_savename = copy.copy(safe_savename_base)
while os.path.isfile(os.path.join(d_save_dir,safe_savename+".npy")):
	repeat_cnt +=1
	safe_savename = "{:}_{:}".format(safe_savename_base,repeat_cnt)

#print(safe_savename)
#sys.exit()

# saving the results, numpy array
with open(os.path.join(d_save_dir,safe_savename+".npy"),"wb") as fid:
	np.save(fid,res_all)
# saving the encoders
# saving the configuration in case of error
with open(os.path.join(d_save_dir,safe_savename+"_config.pkl"),"wb") as fid:
	pickle.dump(argsdict,fid)