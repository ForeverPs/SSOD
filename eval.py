import torch
import numpy as np
from sklearn.metrics import roc_curve, auc


def FPR(id_confidence, id_label, threshold=0.95):
    """ Compute False Positive Recall
    Args:
        id_confidence: 1D numpy array, predicted confidence of sample belonging to ID (in distribution)
        id_label: 1D numpy array, id_label[i] == 1 means sample `i` belong to ID else OOD (out of distribution)
        threshold: float value between 0-1, top `threshold` classified into ID
    Returns:
        fpr_score: False Positive Recall
    """
    id_conf = id_confidence[id_label > 0.5]
    od_conf = id_confidence[id_label < 0.5]
    id_conf = np.sort(id_conf)
    id_threshold = id_conf[int((1 - threshold) * len(id_conf))]

    total_recall_num = np.sum(id_confidence > id_threshold)
    false_recall_num = np.sum(od_conf > id_threshold)
    return false_recall_num / len(od_conf)


def AUROC(id_confidence, id_label):
    """ Compute `The Area Under The Receiver Operating Characteristic Curve`
    Args:
        id_confidence: 1D numpy array, predicted confidence of sample belonging to ID (in distribution)
        id_label: 1D numpy array, id_label[i] == 1 means sample `i` belong to ID else OOD (out of distribution)
    Returns:
        auroc_score: The Area Under The Receiver Operating Characteristic Curve
    """
    fpr, tpr, thresholds = roc_curve(id_label, id_confidence)
    auroc = auc(fpr, tpr)
    return auroc

def ACC(predict, target):
    if predict.shape == target.shape:
        predict = predict.detach().cpu().squeeze().numpy()
        target = target.detach().cpu().squeeze().numpy()
        acc = np.sum(predict == target) / len(predict)
    else:
        predict = torch.argmax(predict, dim=-1)
        predict = predict.detach().cpu().squeeze().numpy()
        target = target.detach().cpu().squeeze().numpy()
        acc = np.sum(predict == target) / len(predict)
    return acc
