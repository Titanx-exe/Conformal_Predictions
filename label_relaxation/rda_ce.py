import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaCompleteAmbiguationPairwiseLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.2, dim=-1, logits_provided=True, one_hot_encode_trgts=True,
                 num_classes=-1, adaptive_beta=False, epochs=None, adaptive_start_beta=None,
                 adaptive_end_beta=None, adaptive_type="linear"):
        super().__init__()
        self.alpha = max(alpha, 1e-3)
        self.beta = beta
        self.dim = dim
        self.logits_provided = logits_provided
        self.one_hot_encode_trgts = one_hot_encode_trgts
        self.num_classes = num_classes

        # Adaptive beta parameters
        self.adaptive_beta = adaptive_beta
        self.epochs = epochs
        self.adaptive_start_beta = adaptive_start_beta
        self.adaptive_end_beta = adaptive_end_beta
        self.adaptive_type = adaptive_type

        if self.adaptive_beta:
            assert epochs is not None and adaptive_start_beta is not None and adaptive_end_beta is not None, \
                "Adaptive beta requires epochs, start and end values."

    def compute_adaptive_beta(self, epoch):
        if self.adaptive_type == "linear":
            return ((1 - epoch / self.epochs) * self.adaptive_start_beta +
                    (epoch / self.epochs) * self.adaptive_end_beta)
        elif self.adaptive_type == "cosine":
            import math
            return self.adaptive_end_beta + 0.5 * (self.adaptive_start_beta - self.adaptive_end_beta) * \
                   (1 + math.cos(math.pi * epoch / self.epochs))
        else:
            raise ValueError(f"Unknown adaptive beta type: {self.adaptive_type}")

    def forward(self, logits, target, epoch=None):
        if self.logits_provided:
            temperature = 20.0  # Try increasing it
            pred = logits / temperature
            pred = F.softmax(pred, dim=self.dim)

        if self.one_hot_encode_trgts:
            target = F.one_hot(target, num_classes=self.num_classes)

        with torch.no_grad():
            # determine current beta
            current_beta = self.compute_adaptive_beta(epoch) if self.adaptive_beta and epoch is not None else self.beta

            # build beta mask
            beta_mask = torch.logical_or(pred > current_beta, target.bool())

            # force ambiguity if ≥2 entries pass threshold
            multi_mask = (beta_mask.sum(dim=-1, keepdim=True) > 1)
            beta_mask = torch.where(multi_mask, torch.ones_like(beta_mask), beta_mask)

            beta_sum = (beta_mask.float() * pred).sum(dim=-1)
            non_beta_sum = ((~beta_mask).float() * pred).sum(dim=-1)

            beta_preds = torch.where(beta_sum.unsqueeze(-1) == 0, torch.zeros_like(pred),
                                     pred / beta_sum.unsqueeze(-1))
            non_beta_preds = torch.where(non_beta_sum.unsqueeze(-1) == 0, torch.zeros_like(pred),
                                         pred / non_beta_sum.unsqueeze(-1))

            target_credal = torch.where(beta_mask, (1 - self.alpha) * beta_preds,
                                        self.alpha * non_beta_preds)

        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)
        is_in_credal_set = beta_sum >= 1.0 - self.alpha
        return torch.where(is_in_credal_set, torch.zeros_like(divergence), divergence).mean()
