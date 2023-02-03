import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat

argv = sys.argv

# argv[1] should be the .npy file
if not ".npy" in argv[1]:
	sys.exit("please enter the .npy full path")
with open(argv[1],"rb") as fid:
	result_array = np.load(fid)

filename = ".".join(argv[1].split(".")[:-1])
with open(filename+"_config.pkl","rb") as fid:
	pkl_config = pickle.load(fid)
print(pkl_config)
# OLD format
# deterministic 
# header
# gamma, nidx, niter, conv,nz, entz, mizx1,mizx2,cmizx1cx2, cmizx2cx1, loss, cmix1x2cz

# stochastic
# header
# # gamma, nidx, niter, conv,nz, entz, mizx1,mizx2,cmizx1cx2, cmizx2cx1, loss, cmix1x2cz

# NEW format
# for other methods
# header
# # gamma, nidx, niter, conv,nz, entz, mizx1,mizx2,joint_mi, loss, cmix1x2cz
header = ["gamma","nidx","niter",'conv','nz',"entz",'mizx1','mizx2','jointMI',"loss",'condMI']
extracted_list = ['gamma','niter','conv','nz','entz','mizx1','mizx2','jointMI','condMI']
# extract gamma,niter,nz,entz,mizx1,mizx2,joint_mi, cmix1x2cz
# and put it into matlab array
# for wyner var form
# nidx, niter, conv,nz, entz, mizx1,mizx2,joint_MI, dkl_error, cmix1x2cz
if pkl_config['method'] == "wyner":
	header = ["nidx",'niter','conv','nz','entz','mizx1','mizx2','jointMI','dklError','condMI']
	extracted_list = ['niter','conv','nz','entz','mizx1','mizx2','jointMI','dklError','condMI']

extract_result = []
for idx in range(result_array.shape[0]):
	tmp_list =[]
	for item in extracted_list:
		tmp_list.append(result_array[idx,header.index(item)])
	extract_result.append(tmp_list)
extract_result = np.array(extract_result).astype("float64")
# reuse the filename
# create converted mat file
os.makedirs("converted_mat",exist_ok=True)
last_fname = os.path.split(argv[1])[-1]
mat_name = ".".join(last_fname.split(".")[:-1])
savemat(os.path.join("converted_mat",mat_name+".mat"),{"result_array":extract_result,"method":pkl_config['method']})
#print(last_fname)
#savemat(os.path.join("converted_mat",filename+".mat"),{"result_array":extract_result,"method":pkl_config['method']})