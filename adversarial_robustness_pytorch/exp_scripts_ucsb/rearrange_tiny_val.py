import os
import pandas as pd
import shutil
import pdb

path = '/scratch/zx1673/dataset/tiny-imagenet'



label_file = os.path.join(path,'val','val_annotations.txt')
imgs_file =  os.path.join(path,'val','images')
new_path = os.path.join(path,'val_new')

if os.path.exists(new_path) == False:
    os.makedirs(new_path)


# make map table
class_dir = os.listdir(os.path.join(path,'train'))
label_data = pd.read_table(label_file)
last_data = label_data.columns.tolist()
label_data.columns = ['image','class','a','b','c','d']
label_data = pd.concat([label_data, pd.DataFrame([last_data], columns=label_data.columns)])
label_data = label_data[['image','class']].values.tolist()


for c in class_dir:
    class_path = os.path.join(new_path,c)
    if os.path.exists(class_path) == False:
        os.makedirs(class_path)

for i in label_data:
    img = i[0]
    lab = i[1]
    ori_file = os.path.join(imgs_file,img)
    tar_file = os.path.join(new_path,lab,img)
    shutil.copy2(ori_file, tar_file)

# class_dir = 