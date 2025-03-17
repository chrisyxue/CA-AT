"""
This is a pipeline for representation anaysis
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import copy


def rep_inside_class_metrics(rep_all,class_centers,y_all,num_class,num_sample):
    """
    Analysis the Metrics inside each class
    """
    L2_dis_matrix = torch.norm((rep_all.unsqueeze(1).repeat([1,num_class,1])  - class_centers.unsqueeze(0).repeat([num_sample,1,1])),p=2,dim=-1) # [num_samples,num_class]
    cos_matrix = F.cosine_similarity(rep_all.unsqueeze(1).repeat([1,num_class,1]),class_centers.unsqueeze(0).repeat([num_sample,1,1]),dim=-1) # [num_samples,num_class]
    Linf_dis_matrix = torch.norm((rep_all.unsqueeze(1).repeat([1,num_class,1])  - class_centers.unsqueeze(0).repeat([num_sample,1,1])),float('inf'),dim=-1)
    y_one_hot_matrix = F.one_hot(y_all)
    # Inside Class Metrics
    ins_class_metrics = {}
    ins_class_metrics['L2_Distance'] = []
    ins_class_metrics['Linf_Distance'] = []
    ins_class_metrics['Cos'] = []
    for c in range(num_class):
        ins_class_metrics['L2_Distance'].append((L2_dis_matrix*y_one_hot_matrix)[y_all==c].mean().item())
        ins_class_metrics['Linf_Distance'].append((Linf_dis_matrix*y_one_hot_matrix)[y_all==c].mean().item())
        ins_class_metrics['Cos'].append((cos_matrix*y_one_hot_matrix)[y_all==c].mean().item())
    return ins_class_metrics


def rep_inter_class_metrics(rep_all,class_centers,y_all,num_class,num_sample):
    """
    Analysis the Metrics cross each class
    """
    L2_dis_matrix = torch.norm((rep_all.unsqueeze(1).repeat([1,num_class,1])  - class_centers.unsqueeze(0).repeat([num_sample,1,1])),p=2,dim=-1) # [num_samples,num_class]
    cos_matrix = F.cosine_similarity(rep_all.unsqueeze(1).repeat([1,num_class,1]),class_centers.unsqueeze(0).repeat([num_sample,1,1]),dim=-1) # [num_samples,num_class]
    Linf_dis_matrix = torch.norm((rep_all.unsqueeze(1).repeat([1,num_class,1])  - class_centers.unsqueeze(0).repeat([num_sample,1,1])),float('inf'),dim=-1)
    y_one_hot_matrix = F.one_hot(y_all)
    # Inter Class Metrics
    inter_class_metrics = {}
    inter_class_metrics['L2_Distance'] = []
    inter_class_metrics['Linf_Distance'] = []
    inter_class_metrics['Cos'] = []
    inver_y_one_hot_matrix = 1 - y_one_hot_matrix
    for c in range(num_class):
        inter_class_metrics['L2_Distance'].append((L2_dis_matrix*inver_y_one_hot_matrix)[y_all==c].mean().item())
        inter_class_metrics['Linf_Distance'].append((Linf_dis_matrix*inver_y_one_hot_matrix)[y_all==c].mean().item())
        inter_class_metrics['Cos'].append((cos_matrix*inver_y_one_hot_matrix)[y_all==c].mean().item())
    return inter_class_metrics




def rep_class_wise_analysis(rep_all,y_all):
    """
    Analysis the properties for a set of representations with labels

    rep_all: Tensor->[num_sample,dim]
    y_all: Tensor->[num_sample]

    Including the L2/Linf/Cos for representations and their related/non-related class center
    """

    num_class = len(torch.unique(y_all).tolist())
    num_sample = rep_all.shape[0]
    class_centers = []
    class_var = []
    for c in range(num_class):
        class_centers.append(rep_all[y_all==c].mean(0,keepdim=True))
        class_var.append(torch.var(rep_all[y_all==c]).item())
    class_centers = torch.concat(class_centers,dim=0)
    
    # Inside Class Metrics
    ins_class_metrics = rep_inside_class_metrics(rep_all,class_centers,y_all,num_class,num_sample)
    # Inter Class Metrics
    inter_class_metrics = rep_inter_class_metrics(rep_all,class_centers,y_all,num_class,num_sample)
    
    return ins_class_metrics, inter_class_metrics, class_var, class_centers




def two_group_rep_class_wise_analysis(rep_all_a,rep_all_b,y_all):
    """
    Analysis the properties for two set of representations/views of a same dataset with their labels

    rep_all_a: Tensor->[num_sample,dim]
    rep_all_b: Tensor->[num_sample,dim]
    y_all: Tensor->[num_sample]

    Including the L2/Linf/Cos for representations and their related/non-related class center
    """
    num_class = len(torch.unique(y_all).tolist())
    num_sample = rep_all_a.shape[0]
    assert(rep_all_a.shape[0] == rep_all_b.shape[0])
    ins_class_metrics_a, inter_class_metrics_a, class_var_a, class_centers_a = rep_class_wise_analysis(rep_all_a,y_all)
    ins_class_metrics_b, inter_class_metrics_b, class_var_b, class_centers_b = rep_class_wise_analysis(rep_all_b,y_all)
    
    y_one_hot_matrix = F.one_hot(y_all)
    
    ## rep a for class center b
    ins_class_metrics_a_b = rep_inside_class_metrics(rep_all_a,class_centers_b,y_all,num_class,num_sample)
    inter_class_metrics_a_b = rep_inter_class_metrics(rep_all_a,class_centers_b,y_all,num_class,num_sample)

    ## rep b for class center a
    ins_class_metrics_b_a = rep_inside_class_metrics(rep_all_b,class_centers_a,y_all,num_class,num_sample)
    inter_class_metrics_b_a = rep_inter_class_metrics(rep_all_b,class_centers_a,y_all,num_class,num_sample)

    return ins_class_metrics_a_b, inter_class_metrics_a_b, ins_class_metrics_b_a, inter_class_metrics_b_a, class_var_a, class_var_b

    



# def logit_class_wise_analysis(logit_all_b,logit_all_a,y_all):
#     """
#     Analysis the properties for a set of logits with labels

#     rep_all: Tensor->[num_sample,num_classes]
#     y_all: Tensor->[num_sample]
#     """

#     num_class = len(torch.unique(y_all).tolist())
#     num_sample = logit_all.shape[0]
#     y_one_hot_matrix = F.one_hot(y_all)
