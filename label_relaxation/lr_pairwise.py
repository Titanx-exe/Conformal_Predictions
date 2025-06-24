import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelRelaxationPairwiseLoss(nn.Module):
    def __init__(self, alpha=0.1, dim=-1, logits_provided=True, one_hot_encode_trgts=True, num_classes=-1):
        super(LabelRelaxationPairwiseLoss, self).__init__()
        self.alpha = alpha
        self.dim = dim

        self.gz_threshold = 0.1
        self.logits_provided = logits_provided
        self.one_hot_encode_trgts = one_hot_encode_trgts
        self.num_classes = num_classes  # should be set to K = num candidates per query

    def forward(self, pred, target):
        """
        pred: Tensor of shape (B, K) — scores/logits for each candidate
        target: Tensor of shape (B,) — index of the positive candidate per query
        """

        if self.logits_provided:
            temperature = 20.0
            pred = pred / temperature
            pred = F.softmax(pred, dim=self.dim)

        if self.one_hot_encode_trgts:
            target = F.one_hot(target, num_classes=self.num_classes).float()

        print("$$$$$$$$$$$$$$$$ Label relaxation parameter for pairwise $$$$$$$$$$$$$$$$$$$$$")
        print(self.alpha)

        with torch.no_grad():
            sum_y_hat_prime = torch.sum((1.0 - target) * pred, dim=self.dim)  # shape: (B,)
            pred_hat = self.alpha * pred / sum_y_hat_prime.unsqueeze(-1)     # shape: (B, K)
            target_credal = torch.where(
                target > self.gz_threshold,
                1.0 - self.alpha,
                pred_hat
            )

        # KL divergence between softened predictions and relaxed label distribution
        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)

        # Confidence on the correct (positive) class
        confidence_on_pos = torch.sum(pred * target, dim=-1)

        # Loss: ignore confident predictions
        result = torch.where(confidence_on_pos > (1.0 - self.alpha), torch.zeros_like(divergence), divergence)

        return result.mean()
