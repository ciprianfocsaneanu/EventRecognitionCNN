import os
import sys
import h5py
from scipy import ndimage
import numpy as np
from PIL import Image

if len(sys.argv) < 2:
    print ("Usage: python gen_h5.py input_folder")
    exit(1)

# Input folder
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
		img_path = fi + cls + '/' + img
		img_aux = Image.open(img_path)
		new_img = img_aux.resize((224, 224))
		os.remove(img_path)
		new_img.save(img_path, "JPEG", optimize=True)
		im = ndimage.imread(img_path)
		set_x.append(im)
		set_y.append(k)
	k +=1

# Sets to numpy arrays
set_x = np.array(set_x)
set_y = np.array(set_y)

# Shuffle sets
rp = np.random.permutation(set_x.shape[0])
set_x = set_x[rp,:]
set_y = set_y[rp]

# Divide sets, train or valid
train_nr_images = 10800
valid_set_x = set_x[0:train_nr_images,:]
valid_set_y = set_y[0:train_nr_images]
train_set_x = set_x[train_nr_images:,:]
train_set_y = set_y[train_nr_images:]

# Convert classes list to ASCII
list_classes = [np.string_(i) for i in list_classes]

# Save h5 files
f = h5py.File('wider-data.h5','w')
f.create_dataset('train_set_x', data=train_set_x, compression='gzip', compression_opts=9)
f.create_dataset('train_set_y', data=train_set_y, compression='gzip', compression_opts=9)
f.create_dataset('valid_set_x', data=valid_set_x, compression='gzip', compression_opts=9)
f.create_dataset('valid_set_y', data=valid_set_y, compression='gzip', compression_opts=9)
f.create_dataset('list_classes', data=list_classes, compression='gzip', compression_opts=9)