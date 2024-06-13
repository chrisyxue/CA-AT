from sklearn.manifold import TSNE
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir('/localscratch/xuezhiyu/adver-robust/Once-for-All-Adversarial-Training/Vis_Res/cifar10/ResNet34/untargeted-pgd-8-7_lambda0.0')
res_lst = []
color_lst = []
type_res = 'fm4'
lst = os.listdir(type_res)
for i in lst:
    if 'dif_0' in i:
        # print(i)
        res = np.load(os.path.join(type_res,i),allow_pickle=True)
        res = res.reshape([-1])
        # res_lst.append(res)
        res_lst.append(res)
        color_lst.append('r')

os.chdir('/localscratch/xuezhiyu/adver-robust/Once-for-All-Adversarial-Training/Vis_Res/cifar10/ResNet34/untargeted-pgd-8-7_lambda0.2')
lst = os.listdir(type_res)
for i in lst:
    if 'dif_0' in i:
        # print(i)
        res = np.load(os.path.join(type_res,i),allow_pickle=True)
        res = res.reshape([-1])
        # res_lst.append(res)
        res_lst.append(res)
        color_lst.append('g')

os.chdir('/localscratch/xuezhiyu/adver-robust/Once-for-All-Adversarial-Training/Vis_Res/cifar10/ResNet34/untargeted-pgd-8-7_lambda0.4')
lst = os.listdir(type_res)
for i in lst:
    if 'dif_0' in i:
        # print(i)
        res = np.load(os.path.join(type_res,i),allow_pickle=True)
        res = res.reshape([-1])
        # res_lst.append(res)
        res_lst.append(res)
        color_lst.append('b')

os.chdir('/localscratch/xuezhiyu/adver-robust/Once-for-All-Adversarial-Training/Vis_Res/cifar10/ResNet34/untargeted-pgd-8-7_lambda0.6')
lst = os.listdir(type_res)
for i in lst:
    if 'dif_0' in i:
        # print(i)
        res = np.load(os.path.join(type_res,i),allow_pickle=True)
        res = res.reshape([-1])
        # res_lst.append(res)
        res_lst.append(res)
        color_lst.append('black')

res_lst = np.array(res_lst)
res_embedded = TSNE(n_components=2, learning_rate='auto',
             init='random').fit_transform(res_lst)

plt.scatter(res_embedded[:,0], res_embedded[:,1],color=color_lst)
plt.savefig('/localscratch/xuezhiyu/adver-robust/Once-for-All-Adversarial-Training/img.png')
print(res_embedded.shape)
