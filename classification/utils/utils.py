from utils.subtype_enum import OVCAREEnum
from utils.subtype_enum import TCGAEnum
from pynvml import *
import glob
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import os


def print_per_class_accuracy(per_class_acc, overall_acc, overall_kappa, overall_auc, overall_f1):
    per_class_acc = per_class_acc * 100
    print('|{:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%|{:.4f}|{:.4f}|{:.4f}|{:.2f}%|'.format(
        per_class_acc[0], per_class_acc[1], per_class_acc[2], per_class_acc[3], per_class_acc[4],  overall_acc * 100, overall_kappa, overall_auc, overall_f1, per_class_acc.mean()))


def compute_metric(labels, preds, probs=None, verbose=False):
    """Function to compute the various metrics given predicted labels and ground truth labels
    Parameters
    ----------
    labels : numpy array
        A row contains the ground truth labels
    preds: numpy array
        A row contains the predicted labels
    probs: numpy array
        A matrix and each row is the probability for the predicted patches or slides
    verbose : bool
        Print detail of the computed metrics
    Returns
    -------
    overall_acc : float
        Accuracy
    overall_kappa : float
        Cohen's kappa
    overall_f1 : float
        F1 score
    overall_auc : float
        ROC AUC
    """
    overall_acc = accuracy_score(labels, preds)
    overall_kappa = cohen_kappa_score(labels, preds)
    overall_f1 = f1_score(labels, preds, average='macro')
    conf_mat = confusion_matrix(labels, preds).T
    acc_per_subtype = conf_mat.diagonal()/conf_mat.sum(axis=0)
    if not (probs is None):
        overall_auc = roc_auc_score(
            labels, probs, multi_class='ovo', average='macro')
    # disply results
    if verbose:
        print('Acc: {:.2f}\%'.format(overall_acc * 100))
        print('Kappa: {:.4f}'.format(overall_kappa))
        print('F1: {:.4f}'.format(overall_f1))
        if not (probs is None):
            print('AUC ROC: {:.4f}'.format(overall_auc))
            print_per_class_accuracy(
                acc_per_subtype, overall_acc, overall_kappa, overall_auc, overall_f1)

    # return results
    if not (probs is None):
        return overall_acc, overall_kappa, overall_f1, overall_auc
    else:
        return overall_acc, overall_kappa, overall_f1, None


def get_label_by_patch_id(patch_id, is_tcga=False):
    """Function to obtain label from patch id

    Parameters
    ----------
    patch_id : string

    Returns
    -------
    label: int
        Integer label from OVCAREEnum or TCGAEnum

    """
    label_idx = -3
    label = patch_id.split('/')[label_idx]
    if is_tcga:
        label = TCGAEnum[label.upper()]
    else:
        label = OVCAREEnum[label.upper()]
    return label.value


def read_data_ids(data_id_path):
    """Function to read data ids (i.e., any *.txt contains row based information)

    Parameters
    ----------
    data_id_path : string
        Absoluate path to the *.txt contains data ids

    Returns
    -------
    data_ids : list
        List conntains data ids

    """
    with open(data_id_path) as f:
        data_ids = f.readlines()
        data_ids = [x.strip() for x in data_ids]
    return data_ids


def set_gpus(n_gpus, verbose=False):
    """Function to set the exposed GPUs

    Parameters
    ----------
    n_gpus : int
        Number of GPUs

    Returns
    -------
    None

    """
    selected_gpu = []
    gpu_free_mem = {}

    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_usage = nvmlDeviceGetMemoryInfo(handle)
        gpu_free_mem[i] = mem_usage.free
        if verbose:
            print("GPU: {} \t Free Memory: {}".format(i, mem_usage.free))

    res = sorted(gpu_free_mem.items(), key=lambda x: x[1], reverse=True)
    res = res[:n_gpus]
    selected_gpu = [r[0] for r in res]

    print("Using GPU {}".format(','.join([str(s) for s in selected_gpu])))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        [str(s) for s in selected_gpu])
