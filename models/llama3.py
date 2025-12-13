"""
Llama3Ranker: Dense retrieval using Llama-3-8B decoder model
Based on E5Ranker and Qwen3Ranker architecture
Uses weighted average pooling for embedding generation
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from label_relaxation import lr_torch, lr_pairwise, lr_beta, rda_ce
from baseline_loss_functions import GCELoss, NCELoss, AUELoss, EvidenceSmoothingLoss, AGCELoss, NCE_AGCELoss, NCE_AUELoss

def _row_minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalization per row"""
    x_min, _ = x.min(dim=1, keepdim=True)
    x_max, _ = x.max(dim=1, keepdim=True)
    denom = (x_max - x_min).clamp_min(eps)
    out = (x - x_min) / denom
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def _soft_label_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft label cross-entropy loss"""
    logp = F.log_softmax(logits, dim=-1)
    return -(targets * logp).sum(dim=1).mean()

class Llama3Ranker(torch.nn.Module):
    """
    Dense retrieval ranker using Llama-3-8B decoder model.

    Key differences from encoder models:
    - Uses causal LM architecture (AutoModelForCausalLM)
    - Employs weighted average pooling considering attention mask
    - Supports gradient checkpointing for memory efficiency
    - Uses left padding for batch inference
    """

    def __init__(self, device=None, params=None,
                 pos_lambda: float = 0.001,
                 neg_lambda: float = 0.01,
                 alpha: float = 0.001,
                 margin: float = 1):
        super(Llama3Ranker, self).__init__()

        self.params = params if params is not None else {}
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin

        # Load Llama-3-8B model
        print("LOADING LLAMA-3.2-1B MODEL FOR DENSE RETRIEVAL")
        model_id = "meta-llama/Llama-3.2-1B"

        # CRITICAL FIX: Initialize tokenizer first (was missing in original)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Set padding configuration for decoder models
        self.tokenizer.padding_side = 'left'  # CRITICAL for causal LM batch inference
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # CRITICAL FIX: Load model to specific device, not device_map="auto"
        # This prevents memory conflicts and ensures proper gradient flow
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
            use_cache=False,  # Disable KV cache for training (critical for memory)
        )

        # Set pad_token_id in model config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Enable gradient checkpointing to save memory (optional but recommended)
        if params and params.get('gradient_checkpointing', True):
            self.model.gradient_checkpointing_enable()

        # Move model to device AFTER loading (not using device_map)
        self.model.to(self.device)

        print(f"✓ Llama3Ranker initialized on {self.device}")
        print(f"  Model dtype: {next(self.model.parameters()).dtype}")
        print(f"  Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")

    def weighted_average_pool(self, last_hidden_states: Tensor,
                               attention_mask: Tensor) -> Tensor:
        """
        Weighted average pooling for decoder models.

        This implementation follows the Stack Overflow reference code:
        - Assigns increasing weights to tokens (1, 2, 3, ...)
        - Masks padding tokens
        - Computes weighted average

        Args:
            last_hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        # Create weights: [1, 2, 3, ..., seq_len] for each position
        seq_len = last_hidden_states.shape[1]
        weights = torch.arange(start=1, end=seq_len + 1, dtype=last_hidden_states.dtype, device=last_hidden_states.device)
        weights = weights.unsqueeze(0)  # [1, seq_len]

        # CRITICAL FIX: Cast attention_mask to same dtype as weights for proper masking
        attention_mask_float = attention_mask.float()

        # Apply attention mask to weights (zero out padding positions)
        weights_for_non_padding = attention_mask_float * weights  # [batch_size, seq_len]

        # Compute weighted sum of embeddings
        # Expand weights to match hidden dimension: [batch_size, seq_len, 1]
        weights_expanded = weights_for_non_padding.unsqueeze(-1)
        sum_embeddings = torch.sum(
            last_hidden_states * weights_expanded,
            dim=1
        )  # [batch_size, hidden_dim]

        # Compute sum of weights (number of non-padding tokens weighted)
        num_of_none_padding_tokens = torch.sum(
            weights_for_non_padding,
            dim=-1
        ).unsqueeze(-1)  # [batch_size, 1]

        # Compute weighted average (avoid division by zero)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens.clamp(min=1e-9)

        return sentence_embeddings

    def encode(self, batch):
        """
        Forward pass through Llama-3 model.

        Args:
            batch: Dictionary with 'input_ids' and 'attention_mask'

        Returns:
            Model outputs with hidden states
        """
        # CRITICAL: Ensure batch is on correct device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Ensure output_hidden_states=True
        return self.model(
            **batch,
            output_hidden_states=True,
            return_dict=True
        )

    def encode_context(self, batch):
        """Encode context/query - delegates to encode_candidate"""
        return self.encode_candidate(batch)

    def encode_candidate(self, batch):
        """
        Encode documents/candidates and return embeddings.

        Args:
            batch: Dictionary with 'input_ids' and 'attention_mask'

        Returns:
            embeddings: [batch_size, hidden_dim] on CPU
        """
        outputs = self.encode(batch)

        # Get last hidden state from decoder
        # For causal LM, hidden_states[-1] is the final layer output
        last_hidden_state = outputs.hidden_states[-1]

        # Apply weighted average pooling
        embeddings = self.weighted_average_pool(
            last_hidden_state,
            batch['attention_mask']
        )

        # Return on CPU for indexing/storage
        return embeddings.cpu().detach()

    def forward_diag(self, input):
        """
        Forward pass for diagonal (in-batch) contrastive learning.
        Queries and documents are concatenated in a single batch.
        """
        EPOCH_FILE = "epoch.txt"
        try:
            with open(EPOCH_FILE, "r") as f:
                epoch = int(f.read().strip())
        except:
            epoch = 0

        # Encode concatenated batch
        outputs = self.encode(input)
        last_hidden_state = outputs.hidden_states[-1]
        embeddings = self.weighted_average_pool(last_hidden_state, input['attention_mask'])

        # Split into query and document embeddings
        context_embed, document_embed = torch.split(
            embeddings,
            int(embeddings.size(0) / 2)
        )

        # Compute similarity scores
        scores = (context_embed @ document_embed.T)

        # Create diagonal targets
        bs = scores.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(self.device)

        # Apply various loss functions based on params
        if self.params.get('label_relaxation_pairwise') == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = lr_pairwise.LabelRelaxationPairwiseLoss(
                alpha=float(self.params['relaxation_param']),
                dim=-1,
                logits_provided=True,
                one_hot_encode_trgts=True,
                num_classes=num_classes
            )
            loss = loss_fn(scores, target)
            print('relaxation parameter applied:', self.params['relaxation_param'])
            return loss, scores

        if self.params.get('ambiguation_loss') == 'yes':
            batch_size, num_classes = scores.size()
            num_epochs = self.params['num_epochs']
            loss_fn = rda_ce.BetaCompleteAmbiguationPairwiseLoss(
                alpha=0.1, beta=0.2, num_classes=num_classes,
                adaptive_beta=True,
                epochs=num_epochs,
                adaptive_start_beta=0.5,
                adaptive_end_beta=0.1,
                adaptive_type="linear"
            )
            loss = loss_fn(scores, target, epoch=epoch + 1)
            return loss, scores

        if self.params.get('gce_loss') == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = GCELoss.GCELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('nce_loss') == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = NCELoss.NCELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('nce_agce') == 'yes':
            print("NCE AGCE loss applied")
            batch_size, num_classes = scores.size()
            loss_fn = NCE_AGCELoss.NCEandAGCE(num_classes=num_classes)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('aue_loss') == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = AUELoss.AUELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('mbls') == 'yes':
            return self.mbls_forward(scores, target)

        if self.params.get('acls') == 'yes':
            return self.acls_forward(scores, target)

        # Default: cross-entropy loss
        loss = F.cross_entropy(scores, target, reduction="mean")
        return loss, scores

    def wsls_forward(self, scores: torch.Tensor, target: torch.Tensor,
                     ns_scores: torch.Tensor = None):
        """Weakly Supervised Label Smoothing"""
        print("Applying weakly supervised loss (Llama3)")
        B, C = scores.shape
        device = scores.device
        row_idx = torch.arange(B, device=device)

        eps_wsls = float(self.params.get('wsls_eps', 0.2))
        try:
            with open("epoch.txt", "r") as f:
                cur_epoch = int(f.read().strip())
        except Exception:
            cur_epoch = 0

        if self.params.get('wsls_two_stage', 'no') == 'yes':
            switch_ep = int(self.params.get('wsls_switch_epoch', max(cur_epoch + 1, 1)))
            if cur_epoch >= switch_ep:
                eps_wsls = 0.0

        tau = float(self.params.get('wsls_tau', 1.0))
        s = scores.detach()
        if tau != 1.0:
            s = s / tau

        s_neg = s.clone()
        row_min = s_neg.min(dim=1, keepdim=True).values
        s_neg[row_idx, target] = (row_min.squeeze(1) - 1.0)

        w = _row_minmax_norm(s_neg)
        w[row_idx, target] = 1.0 / C
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)

        one_hot = F.one_hot(target, num_classes=C).float()
        tgt_soft = (1.0 - eps_wsls) * one_hot + eps_wsls * w

        loss = _soft_label_ce(scores, tgt_soft)
        return loss, scores

    def get_reg(self, inputs, targets):
        """Adaptive Confidence Loss Smoothing regularization"""
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        indicator = (max_values.clone().detach() == inputs.clone().detach()).float()

        batch_size, num_classes = inputs.size()
        num_pos = batch_size * 1.0
        num_neg = batch_size * (num_classes - 1.0)

        neg_dist = max_values.clone().detach() - inputs
        pos_dist_margin = F.relu(max_values - self.margin)
        neg_dist_margin = F.relu(neg_dist - self.margin)

        pos = indicator * pos_dist_margin ** 2
        neg = (1.0 - indicator) * (neg_dist_margin ** 2)

        reg = self.pos_lambda * (pos.sum() / num_pos) + self.neg_lambda * (neg.sum() / num_neg)
        return reg

    def acls_forward(self, scores, targets, alpha=0.1):
        """Adaptive Confidence Loss Smoothing"""
        if scores.dim() > 2:
            scores = scores.view(scores.size(0), scores.size(1), -1)
            scores = scores.transpose(1, 2)
            scores = scores.contiguous().view(-1, scores.size(2))
            targets = targets.view(-1)

        loss_ce = F.cross_entropy(scores, targets, reduction="mean")
        print("ACLS applied (Llama3)")
        loss_reg = self.get_reg(scores, targets)
        loss = loss_ce + self.alpha * loss_reg

        return loss, scores

    def get_diff(self, scores):
        """Compute difference from max scores"""
        max_values = scores.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, scores.shape[1])
        diff = max_values - scores
        return diff

    def mbls_forward(self, scores, target):
        """Margin-Based Label Smoothing"""
        alpha = 0.1
        margin = 10

        if scores.dim() > 2:
            scores = scores.view(scores.size(0), scores.size(1), -1)
            scores = scores.transpose(1, 2)
            scores = scores.contiguous().view(-1, scores.size(2))
            targets = target.view(-1)

        loss_ce = F.cross_entropy(scores, target, reduction='mean')
        diff = self.get_diff(scores)
        loss_margin = F.relu(diff - margin).mean()
        loss = loss_ce + alpha * loss_margin

        return loss, scores

    def per_example_calibration_error(self, logits, labels):
        """
        Compute calibration error per example.
        
        Args:
            logits: Similarity scores [batch_size, num_candidates]
            labels: Ground truth labels [batch_size]
        
        Returns:
            Per-example calibration error [batch_size]
        """
        # CRITICAL FIX: Handle edge cases with empty or malformed tensors
        if logits.dim() < 2:
            # If logits is 1D or 0D, return zeros
            return torch.zeros(logits.size(0) if logits.dim() > 0 else 0, device=logits.device)
        
        if logits.size(0) == 0 or logits.size(1) == 0:
            # If batch size is 0 or num_classes is 0, return empty tensor
            return torch.zeros(logits.size(0), device=logits.device)
        
        # Normal computation
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        correctness = predictions.eq(labels).float()
        return torch.abs(confidences - correctness)

    def compute_recall_at_k(self, scores, k=5):
        """Compute Recall@k"""
        topk = torch.topk(scores, k=k, dim=1).indices
        targets = torch.arange(scores.size(0)).unsqueeze(1).to(scores.device)
        match = (topk == targets).any(dim=1).float()
        recall_at_k = match.mean().item()
        return recall_at_k

    def forward(self, doc_input, context_len=None, target=None):
        """
        Main forward pass for training.

        Args:
            doc_input: Tokenized input (queries + documents concatenated)
            context_len: Number of queries in the batch
            target: Ground truth labels (diagonal indices)

        Returns:
            loss: Computed loss
            scores: Similarity matrix [num_queries, num_documents]
        """
        EPOCH_FILE = "epoch.txt"
        try:
            with open(EPOCH_FILE, "r") as f:
                epoch = int(f.read().strip())
        except:
            epoch = 0

        # If no context_len provided, use diagonal mode
        if context_len is None:
            return self.forward_diag(doc_input)

        # Encode entire batch
        output = self.encode(doc_input)
        last_hidden_state = output.hidden_states[-1]
        embeddings = self.weighted_average_pool(last_hidden_state, doc_input['attention_mask'])

        # CRITICAL FIX: Verify dimensions before split
        total_size = embeddings.size(0)
        doc_len = total_size - context_len

        
        if doc_len <= 0:
            raise ValueError(f"Invalid split: total_size={total_size}, context_len={context_len}")

        # Split into context (queries) and candidates (documents)
        # CRITICAL FIX: Use explicit slicing instead of torch.split
        context_embeddings = embeddings[:context_len]  # First context_len items
        candidate_embeddings = embeddings[context_len:]  # Remaining items

        
        # Compute similarity scores
        scores = (context_embeddings @ candidate_embeddings.T)

        
        # Create target if not provided
        if target is None:
            bs = scores.size(0)
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)

        
        # CRITICAL FIX: Ensure target matches scores dimensions
        if scores.size(0) != target.size(0):
            print(f"WARNING: Dimension mismatch! scores={scores.shape}, target={target.shape}")
            # Use the minimum dimension to avoid crashes
            min_size = min(scores.size(0), target.size(0))
            target = target[:min_size]
            scores = scores[:min_size, :]
            print(f"ADJUSTED: scores={scores.shape}, target={target.shape}")

        # Compute calibration error (just computes, doesn't use the return value)
        if scores.size(0) > 0 and scores.size(1) > 0:
            try:
                _ = self.per_example_calibration_error(scores, target)
            except Exception as e:
                print(f"Calibration error failed: {e}")
                pass

        # Apply loss functions based on params
        if self.params.get('label_relaxation') == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = lr_torch.LabelRelaxationLoss(
                alpha=float(self.params['relaxation_param']),
                dim=-1,
                logits_provided=True,
                one_hot_encode_trgts=True,
                num_classes=num_classes
            )
            loss = loss_fn(scores, target)
            print("Label relaxation applied (Llama3) with parameter:", self.params['relaxation_param'])
            return loss, scores

        if self.params.get('label_relaxation_pairwise') == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = lr_pairwise.LabelRelaxationPairwiseLoss(
                alpha=float(self.params['relaxation_param']),
                dim=-1,
                logits_provided=True,
                one_hot_encode_trgts=True,
                num_classes=num_classes
            )
            loss = loss_fn(scores, target)
            print("Label relaxation applied (Llama3):", self.params['relaxation_param'])
            return loss, scores

        if self.params.get('ambiguation_loss') == 'yes':
            batch_size, num_classes = scores.size()
            num_epochs = self.params['num_epochs']
            loss_fn = rda_ce.BetaCompleteAmbiguationPairwiseLoss(
                alpha=0.1, beta=0.2, num_classes=num_classes,
                adaptive_beta=True,
                epochs=num_epochs,
                adaptive_start_beta=0.5,
                adaptive_end_beta=0.1,
                adaptive_type="linear"
            )
            loss = loss_fn(scores, target, epoch=epoch + 1)
            recall_5 = self.compute_recall_at_k(scores, k=5)
            return loss, scores

        if self.params.get('gce_loss') == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = GCELoss.GCELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('nce_loss') == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = NCELoss.NCELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('aue_loss') == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = AUELoss.AUELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('nce_aue') == 'yes':
            print("APPLYING NCE AUE LOSS TOGETHER (Llama3)")
            batch_size, num_classes = scores.size()
            loss_fn = NCE_AUELoss.NCEandAUE(num_classes=num_classes)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('agce') == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = AGCELoss.AGCELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('ebls', 'no') == 'yes':
            eps = 0.1
            k = 5
            lam = 0.5
            metric = 'cosine'
            loss, _ = EvidenceSmoothingLoss.ebls_loss_rows(
                scores=scores,
                cand_embs=candidate_embeddings.detach(),
                target=target,
                eps=eps, k=k, lam=lam, metric=metric,
                detach_evidence=True
            )
            return loss, scores

        if self.params.get('wsls', 'no') == 'yes':
            return self.wsls_forward(scores, target)

        if self.params.get('mbls') == 'yes':
            return self.mbls_forward(scores, target)

        if self.params.get('acls') == 'yes':
            return self.acls_forward(scores, target)

        if self.params.get('adaptive_epoch') == 'yes':
            if epoch > self.params.get('epoch_bound', 5):
                self.params['label_smoothness'] = self.params.get('label_smoothness', 0.1) - 1.0

        # Default: cross-entropy loss with optional label smoothing
        label_smoothing = float(self.params.get('label_smoothness', 0.0))
        loss = F.cross_entropy(scores, target, reduction="mean", label_smoothing=label_smoothing)

        return loss, scores

