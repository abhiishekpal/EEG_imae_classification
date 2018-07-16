import os
import sys


def get_label()
    path = '/home/knot/Documents/MTP/eeg_data_cvpr_2017/label.txt'
    lis = []
    lis_label = []
    classes_present = []
    with open(path, 'r') as f:
        for li in f:
            name = li.split(" ")[0]
            label = li[len(name)+1:len(li)-1]
            lis.append(name)
            lis_label.append(label)

    import scipy.io
    path = "/home/knot/Documents/MTP/eeg_data_cvpr_2017/eeg_matlab/6"

    ct = 0;
    for f in os.listdir(path):
        name = f.split("_")[0]
        ind = lis.index(name)
        if(lis_label[ind] in classes_present):
            continue
        ct+=1
        classes_present.append(lis_label[ind])

        print ct,lis_label[ind] 
        
        
    
# print(content)


