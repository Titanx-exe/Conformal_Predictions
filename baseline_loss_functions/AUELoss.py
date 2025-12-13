import torch
import torch.nn as nn
import torch.nn.functional as F


class AUELoss(nn.Module):
    def __init__(self, num_classes=10, a=5.5, q=3., scale=1.0):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q
        #self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a - 1) ** self.q) / self.q
        return loss.mean() * self.scale