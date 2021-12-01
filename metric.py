'''
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
'''
import numpy as np
#from hausdorff import hausdorff_distance
'''
class DiceScore(nn.Module):
    def __init__(self):
        super(DiceScore, self).__init__()

    def forward(self, logits, labels):
        num = labels.size(0)
        predicts = logits.view(num, -1).float()
        labels = labels.view(num, -1).float()
        intersection = (predicts * labels)
        score = 2. * (intersection.sum(1)) / (predicts.sum(1) + labels.sum(1) + 1e-5)
        return score.mean()

'''

def dice_score_list(label_gt, label_pred, n_class):
    """
    :param label_gt: [WxH] (2D images)
    :param label_pred: [WxH] (2D images)
    :param n_class: number of label classes
    :return:
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32).flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            dice_scores[batch_id, class_id] = score

    return np.mean(dice_scores, axis=0)



def dice_score(label_gt, label_pred, n_class):

    """
    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    dice_scores = np.zeros(n_class, dtype=np.float32)
    for class_id in range(n_class):
        img_A = np.array(label_gt == class_id, dtype=np.float32).flatten()
        img_B = np.array(label_pred == class_id, dtype=np.float32).flatten()
        score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
        dice_scores[class_id] = score

    return dice_scores
    
    
def dice_score2(label_gt, label_pred, n_class):

    """
    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    dice_scores = np.zeros(n_class, dtype=np.float32)
#    label_pred[label_pred == 0] = 100
#    label_gt[label_gt == 0] = 100
    for class_id in range(1,n_class):
        l_gt = label_gt.copy()
        l_pred = label_pred.copy()
        l_gt[l_gt==class_id]=100
        l_pred[l_pred==class_id]=100
        img_A = np.array(l_gt == 100, dtype=np.float32).flatten()
        img_B = np.array(l_pred == 100, dtype=np.float32).flatten()
        score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
        dice_scores[class_id] = score

    return dice_scores
def sensitivity(label_gt, label_pred, n_class):

    """
    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    dice_scores = np.zeros(n_class, dtype=np.float32)
#    label_pred[label_pred == 0] = 100
#    label_gt[label_gt == 0] = 100
    for class_id in range(1,n_class):
        l_gt = label_gt.copy()
        l_pred = label_pred.copy()
        l_gt[l_gt==class_id]=100
        l_pred[l_pred==class_id]=100
        img_A = np.array(l_gt == 100, dtype=np.float32).flatten()
        img_B = np.array(l_pred == 100, dtype=np.float32).flatten()
        score = np.sum(img_A * img_B) / (np.sum(img_A)  + epsilon)
        dice_scores[class_id] = score

    return dice_scores
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
