import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pdb

def sort_lst(lst):
    sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
    indices = [i for i, _ in sorted_lst]
    values = [x for _, x in sorted_lst]
    return indices, values

# PATH of the Model
path = '/scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce_grad_track_multiattack/1_resnet18_adver_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4'
path = '/scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce_grad_track_multiattack/1_resnet18_adver_avg_beta_0.5_gradnorm/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4'
path = '/scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce_grad_track_multiattack/1_resnet18_adver_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/'
path = '/scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce_grad_track_multiattack/1_resnet18_adver_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4'
# path = '/scratch/zx1673/results/xuezhiyu/adver_robustness/tiny-imagenet/1_preact-resnet18_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4'
fig_path = os.path.join(path,'fig')
if os.path.exists(fig_path) is False:
    os.makedirs(fig_path)

metric = 'Conflict'
attacks = ['APGD-CE','APGD-DLR','PGD-CE','TAPGD-DLR','T-FAB']
res_all_attack = []
for attk in attacks:
    grad_track_files = [i for i in os.listdir(path) if attk in i]
    if attk == 'APGD-DLR':
        grad_track_files = [i for i in grad_track_files if 'TAPGD' not in i]
    if attk == 'PGD-CE':
        grad_track_files = [i for i in grad_track_files if 'APGD' not in i]
    # pdb.set_trace()
    print([int(i.split('_')[0][len(attk):]) for i in grad_track_files])
    grad_track_idx,grad_track_values = sort_lst([int(i.split('_')[0][len(attk):]) for i in grad_track_files])
    grad_track_files = np.array(grad_track_files)[grad_track_idx].tolist()

    res = []
    epoch = []
    for file in grad_track_files:
        file_path = os.path.join(path,file) 
        df = pd.read_csv(file_path)
        df = df.set_index('Unnamed: 0')
        res.append(df[metric])
        epoch.append(int(file.split('_')[0][len(attk):]))
    res = pd.DataFrame(res).set_index([epoch]) 
    res_all_attack.append(res)

res_all_attack = pd.concat(res_all_attack, axis=1)
res_all_attack.columns = attacks
fig_path_temp = os.path.join(fig_path,'Cos_of_Adver_vs_Clean(Whole)') 
if os.path.exists(fig_path_temp) is False:
    os.makedirs(fig_path_temp)

plt.figure(figsize=[10,5])
res_all_attack.plot(kind='line')
plt.title(metric)
plt.xlabel('Epoch')
plt.savefig(os.path.join(fig_path_temp,metric),bbox_inches = "tight")
pdb.set_trace()



grad_track_layers_files = [i for i in os.listdir(path) if 'grad_track_layers' in i]
grad_track_layers_idx,grad_track_layers_values = sort_lst([int(i.split('_')[0]) for i in grad_track_layers_files])
grad_track_layers_files = np.array(grad_track_layers_files)[grad_track_layers_idx].tolist()
grad_track_files = [i for i in os.listdir(path) if 'grad_track.csv' in i]
grad_track_idx,grad_track_values = sort_lst([int(i.split('_')[0]) for i in grad_track_files])
grad_track_files = np.array(grad_track_files)[grad_track_idx].tolist()

metric = 'Conflict'
metric = 'Cos'
metric =  'Eucli'
metric = 'Clean_L2Norm'
metric = 'Adver_L2Norm'
norm = True

"""
L2 Norm of Adver Gradient and Clean Gradient
"""
metric = 'Clean_L2Norm'
res_clean = []
epoch = []
for file in grad_track_layers_files:
    file_path = os.path.join(path,file) 
    df = pd.read_csv(file_path)
    df = df.set_index('Unnamed: 0')
    res_clean.append(df.loc[metric,:])
    epoch.append(int(file.split('_')[0]))

res_clean = pd.DataFrame(res_clean).set_index([epoch]) 
Dim = df.loc['Dim']
res_clean = res_clean.append(Dim)
res_clean = res_clean.drop(columns=res_clean.columns[res_clean.eq(0).all()], axis=1)
res_clean = (res_clean / res_clean.loc['Dim']).drop(index=['Dim'])



metric = 'Adver_L2Norm'
res_adver = []
epoch = []
for file in grad_track_layers_files:
    file_path = os.path.join(path,file) 
    df = pd.read_csv(file_path)
    df = df.set_index('Unnamed: 0')
    res_adver.append(df.loc[metric,:])
    epoch.append(int(file.split('_')[0]))

res_adver = pd.DataFrame(res_adver).set_index([epoch]) 
Dim = df.loc['Dim']
res_adver = res_adver.append(Dim)
res_adver = res_adver.drop(columns=res_adver.columns[res_adver.eq(0).all()], axis=1)
res_adver = (res_adver / res_adver.loc['Dim']).drop(index=['Dim'])

fig_path_temp = os.path.join(fig_path,'Norm_of_Adver_vs_Clean') 
if os.path.exists(fig_path_temp) is False:
    os.makedirs(fig_path_temp)
for i in epoch:
    plt.figure(figsize=[20,5])
    data_epoch = pd.concat([res_clean.loc[i],res_adver.loc[i],res_adver.loc[i]-res_clean.loc[i]],axis=1,keys=['clean_l2','adver_l2','dif_l2'])
    data_epoch.plot(kind='bar')
    plt.title('Epoch='+str(i))
    plt.savefig(os.path.join(fig_path_temp,'Epoch='+str(i)),bbox_inches = "tight")



"""
Cosine of L2 Norm of Adver Gradient and Clean Gradient
"""
metric = 'Cos'
res_cos = []
epoch = []
for file in grad_track_layers_files:
    file_path = os.path.join(path,file) 
    df = pd.read_csv(file_path)
    df = df.set_index('Unnamed: 0')
    res_cos.append(df.loc[metric,:])
    epoch.append(int(file.split('_')[0]))

res_cos = pd.DataFrame(res_cos).set_index([epoch]) 
res_cos = res_cos.drop(columns=res_cos.columns[res_cos.eq(0).all()], axis=1)

fig_path_temp = os.path.join(fig_path,'Cos_of_Adver_vs_Clean') 
if os.path.exists(fig_path_temp) is False:
    os.makedirs(fig_path_temp)
for i in epoch:
    plt.figure(figsize=[10,5])
    data_epoch = res_cos.loc[i]
    data_epoch.plot(kind='bar')
    plt.title('Epoch='+str(i))
    plt.savefig(os.path.join(fig_path_temp,'Epoch='+str(i)),bbox_inches = "tight")

# plt.figure(figsize=[35,5])
res_cos.T.plot(kind='bar',figsize=[25,5])
# res_cos.plot(kind='line',figsize=[25,5])
plt.savefig(os.path.join(fig_path_temp,'All'),bbox_inches = "tight")
pdb.set_trace()


"""
Cosine of L2 Norm of Adver Gradient and Clean Gradient (Whole Model)
"""
metric = 'Cos'
res_cos = []
epoch = []
for file in grad_track_files:
    file_path = os.path.join(path,file) 
    df = pd.read_csv(file_path)
    df = df.set_index('Unnamed: 0')
    res_cos.append(df[metric])
    epoch.append(int(file.split('_')[0]))

res_cos = pd.DataFrame(res_cos).set_index([epoch]) 

fig_path_temp = os.path.join(fig_path,'Cos_of_Adver_vs_Clean(Whole)') 
if os.path.exists(fig_path_temp) is False:
    os.makedirs(fig_path_temp)
plt.figure(figsize=[10,5])
res_cos.plot(kind='line')
plt.title('Cos')
plt.xlabel('Epoch')
plt.savefig(os.path.join(fig_path_temp,'Cos'),bbox_inches = "tight")


