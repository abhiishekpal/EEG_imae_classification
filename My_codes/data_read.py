import scipy.io
import progressbar
import numpy as np
import os

path = '/home/knot/Documents/MTP/data/eeg_data_cvpr_2017/label.txt'
lis = []
lis_label = []
classes_present = []
with open(path, 'r') as f:
    for li in f:
        name = li.split(" ")[0]
        label = li[len(name)+1:len(li)-1]
        lis.append(name)
        lis_label.append(label)

path = "/home/knot/Documents/MTP/data/eeg_data_cvpr_2017/eeg_matlab/6"
ct = 0;
new_lis = []
new_label = []
for f in os.listdir(path):
    name = f.split("_")[0]
    ind = lis.index(name)
    if(lis_label[ind] in classes_present):
        continue
    classes_present.append(lis_label[ind])

    new_label.append(ct)
    new_lis.append(name)
    ct+=1

total = 0

for i in range(1,6):
    path = "/home/knot/Documents/MTP/data/eeg_data_cvpr_2017/eeg_matlab/{}".format(i)
    for file in os.listdir(path):
        total+=1


print(total)
# X_f = np.ndarray(shape=(total,200,128), dtype = float)
# Y_f = np.ndarray(shape=(total,1,40), dtype = float)

# ct = 0
# for i in range(1,6):
#     path = "/home/knot/Documents/MTP/data/eeg_data_cvpr_2017/eeg_matlab/{}".format(i)
#     for file in progressbar.progressbar(os.listdir(path)):
#         name = file.split("_")[0]
#         ind = new_lis.index(name)
#         label = new_label[ind]
#         mat = scipy.io.loadmat(os.path.join(path,file))
#         X = np.zeros((200,128))
#         X = np.copy(mat['x'][200:400,:])
#         X = np.array(X)
#         Y = np.zeros((1,40))
#         Y[0][label] = 1
#         Y = np.array(Y)
#
#         X_f[ct,:,:] = X
#         Y_f[ct,:,:] = Y
#
# np.save("/home/knot/Documents/MTP/data/input_eeg.npy", X_f)
# np.save("/home/knot/Documents/MTP/data/eeg_label.npy", Y_f)
