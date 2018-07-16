import os
import sys
import scipy.io


path = "/home/knot/Documents/MTP/eeg_data_cvpr_2017/eeg_matlab"

for file in os.listdir(path):
    label = get_label(file)
    mat = scipy.io.loadmat(file)
    
#     print(mat['x'])
    
