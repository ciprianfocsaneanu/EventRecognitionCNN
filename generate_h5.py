import os
import sys
import h5py
from scipy import ndimage
import numpy as np

if len(sys.argv) < 2:
    print ("Usage: python gen_h5.py input_folder")
    exit(1)

# input folder
fi = sys.argv[1]

# init var
classes = os.listdir(fi)
set_x = []
set_y = []
k = 0 # idx for classes
list_classes = []

# Create sets
for cls in classes:
	list_classes.append(cls)
	imgs = os.listdir(fi + cls)
	for img in imgs:
		im = ndimage.imread(fi + cls + '/' + img)
		set_x.append(im)
		set_y.append(k)
	k +=1

# sets to numpy arrays
set_x = np.array(set_x)
set_y = np.array(set_y)

# shuffle sets
rp = np.random.permutation(set_x.shape[0])
set_x = set_x[rp,:]
set_y = set_y[rp]

print (set_x.shape)
print (set_y.shape)
sys.exit()

# divide sets, train or valid
valid_set_x = set_x[0:set_x.shape[0]/10,:]
valid_set_y = set_y[0:set_x.shape[0]/10]
train_set_x = set_x[set_x.shape[0]/10:,:]
train_set_y = set_y[set_x.shape[0]/10:]

# save h5 files
f = h5py.File('data.h5','w')
f.create_dataset('train_set_x', data=train_set_x)
f.create_dataset('train_set_y', data=train_set_y)
f.create_dataset('valid_set_x', data=valid_set_x)
f.create_dataset('valid_set_y', data=valid_set_y)
f.create_dataset('list_classes', data=list_classes)