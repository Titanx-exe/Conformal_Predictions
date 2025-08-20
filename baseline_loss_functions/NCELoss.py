import torch
import torch.nn as nn
import torch.nn.functional as F

class NCELoss(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()