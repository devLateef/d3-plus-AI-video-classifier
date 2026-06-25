"""
loss.py
Self-regularization and feature fusion losses from SWA-DAL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def get_sr_loss(out1, out2, sr_epsilon=0.4, sr_loss_p=0.5):
    """
    Self-Regularization loss.
    Enforces consistency between weak and strong augmentations.
    """
    prob1_t = F.softmax(out1, dim=1)
    prob2_t = F.softmax(out2, dim=1)

    prob1 = F.softmax(out1, dim=1)
    log_prob1 = F.log_softmax(out1, dim=1)
    prob2 = F.softmax(out2, dim=1)
    log_prob2 = F.log_softmax(out2, dim=1)

    if random.random() <= sr_loss_p:
        log_prob2 = F.log_softmax(out2, dim=1)
        mask1 = (prob1_t.max(-1)[0] > sr_epsilon).float()
        aug_loss = ((prob1 * (log_prob1 - log_prob2)).sum(-1) * mask1).sum() / (mask1.sum() + 1e-6)
    else:
        log_prob1 = F.log_softmax(out1, dim=1)
        mask2 = (prob2_t.max(-1)[0] > sr_epsilon).float()
        aug_loss = ((prob2 * (log_prob2 - log_prob1)).sum(-1) * mask2).sum() / (mask2.sum() + 1e-6)

    return aug_loss


def get_ffvit_loss(outs, outt, outst, sr_epsilon=0.4, sr_loss_p=0.5, sr_alpha=0.3):
    """
    Feature Fusion ViT loss.
    Enforces consistency between individual views and fused representation.
    """
    prob_s = F.softmax(outs, dim=1)
    log_prob_s = F.log_softmax(outs, dim=1)
    prob_t = F.softmax(outt, dim=1)
    log_prob_t = F.log_softmax(outt, dim=1)
    prob_st = F.softmax(outst, dim=1)
    log_prob_st = F.log_softmax(outst, dim=1)

    if random.random() <= sr_loss_p:
        mask_s = (prob_s.max(-1)[0] > sr_epsilon).float()
        mask_t = (prob_t.max(-1)[0] > sr_epsilon).float()
        aug_loss = ((prob_s * (log_prob_s - log_prob_st)).sum(-1) * mask_s).sum() / (mask_s.sum() + 1e-6) * sr_alpha
        aug_loss += ((prob_t * (log_prob_t - log_prob_st)).sum(-1) * mask_t).sum() / (mask_t.sum() + 1e-6) * (1 - sr_alpha)
    else:
        mask_st = (prob_st.max(-1)[0] > sr_epsilon).float()
        aug_loss = ((prob_st * (log_prob_st - log_prob_s)).sum(-1) * mask_st).sum() / (mask_st.sum() + 1e-6) * sr_alpha
        aug_loss += ((prob_st * (log_prob_st - log_prob_t)).sum(-1) * mask_st).sum() / (mask_st.sum() + 1e-6) * (1 - sr_alpha)

    return aug_loss