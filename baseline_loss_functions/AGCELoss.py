import torch
import torch.nn as nn
import torch.nn.functional as F


class AGCELoss(nn.Module):
    def __init__(self, num_classes=10, a=0.6, q=0.6, scale=1.):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes
        self.eps = 0.1
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = ((self.a + 1) ** self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean() * self.scale