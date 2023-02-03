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
#print(result_array.shape)
# NEW format
# for representation form
# header
# gamma, nidx, niter, conv,nz, entz, mizx1,mizx2,joint_MI, cmix1x2cz, train_acc, test_acc
# modified after 2023.02.03
header = ["gamma","nidx","niter",'conv','nz',"entz",'mizx1','mizx2','jointMI','condMI','dkl_error',"train_acc",'test_acc']
# and put it into matlab array
extracted_list = ['gamma','niter','conv','nz','entz','mizx1','mizx2','jointMI','condMI','dkl_error',"train_acc",'test_acc']
# For variational form
extract_result = []
for idx in range(result_array.shape[0]):
	tmp_list =[]
	for item in extracted_list:
		tmp_list.append(result_array[idx,header.index(item)])
	extract_result.append(tmp_list)
extract_result = np.array(extract_result).astype("float64")
# reuse the filename
# create converted mat file
os.makedirs("converted_test_mat",exist_ok=True)
last_fname = os.path.split(argv[1])[-1]
mat_name = ".".join(last_fname.split(".")[:-1])
savemat(os.path.join("converted_test_mat",mat_name+".mat"),{"result_array":extract_result,"method":pkl_config['method']})
