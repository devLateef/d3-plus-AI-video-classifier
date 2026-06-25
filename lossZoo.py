"""
lossZoo.py
Additional loss functions for domain generalization.
"""

import torch
import torch.nn as nn
import numpy as np


class ConsistencyCos(nn.Module):
    """
    Cosine consistency loss for feature alignment.
    """
    def __init__(self):
        super(ConsistencyCos, self).__init__()
        self.mse_fn = nn.MSELoss()

    def forward(self, feat):
        feat = nn.functional.normalize(feat, dim=1)
        feat_0 = feat[:int(feat.size(0)/2), :]
        feat_1 = feat[int(feat.size(0)/2):, :]
        cos = torch.einsum('nc,nc->n', [feat_0, feat_1]).unsqueeze(-1)
        labels = torch.ones((cos.shape[0], 1), dtype=torch.float, requires_grad=False)
        if torch.cuda.is_available():
            labels = labels.cuda()
        loss = self.mse_fn(cos, labels)
        return loss


def entropy(input_):
    """Entropy loss for information maximization."""
    epsilon = 1e-10
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def im(outputs_test, gent=True):
    """
    Information Maximization loss.
    Encourages confident predictions while maintaining diversity.
    """
    epsilon = 1e-10
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    entropy_loss = torch.mean(entropy(softmax_out))
    if gent:
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log2(msoftmax + epsilon))
        entropy_loss -= gentropy_loss
    im_loss = entropy_loss * 1.0
    return im_loss