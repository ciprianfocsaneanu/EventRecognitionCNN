import os
from shutil import copyfile

dataset_path = './newWIDER/';

new_dir_path = './h5_to_be/';
if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)

categ_dirs = os.listdir(dataset_path + 'train')
for categ in categ_dirs:
    new_directory = new_dir_path + '/' + categ;
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    for img_file in os.listdir(dataset_path + 'train/' + categ):
        copyfile(dataset_path + 'train/' + categ + '/' + img_file, new_dir_path + categ + '/' + img_file)
    for img_file in os.listdir(dataset_path + 'validation/' + categ):
        copyfile(dataset_path + 'validation/' + categ + '/' + img_file, new_dir_path + categ + '/' + img_file)
