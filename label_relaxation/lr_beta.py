import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaLabelRelaxationPairwiseLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.2, dim=-1, logits_provided=True, one_hot_encode_trgts=True, num_classes=-1):
        super().__init__()
        self.alpha = max(alpha, 1e-3)
        self.beta = beta
        self.dim = dim
        self.logits_provided = logits_provided
        self.one_hot_encode_trgts = one_hot_encode_trgts
        self.num_classes = num_classes

    def forward(self, pred, target):
        if self.logits_provided:
            pred = F.softmax(pred / 20.0, dim=self.dim)

        if self.one_hot_encode_trgts:
            target = F.one_hot(target, num_classes=self.num_classes).float()

        with torch.no_grad():
            beta_mask = torch.logical_or(pred > self.beta, target.bool())
            beta_sum = (beta_mask.float() * pred).sum(dim=-1)
            non_beta_sum = ((~beta_mask).float() * pred).sum(dim=-1)

            beta_preds = torch.where(beta_sum.unsqueeze(-1) == 0, torch.zeros_like(pred), pred / beta_sum.unsqueeze(-1))
            non_beta_preds = torch.where(non_beta_sum.unsqueeze(-1) == 0, torch.zeros_like(pred), pred / non_beta_sum.unsqueeze(-1))

            target_credal = torch.where(beta_mask, (1 - self.alpha) * beta_preds, self.alpha * non_beta_preds)

        divergence = F.kl_div(pred.log(), target_credal, log_target=False, reduction='none').sum(dim=-1)
        is_in_credal_set = beta_sum >= 1.0 - self.alpha
        return torch.where(is_in_credal_set, torch.zeros_like(divergence), divergence).mean()
