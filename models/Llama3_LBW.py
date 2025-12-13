import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoConfig
from models.modeling_llama_lbw import LlamaModelLBW

# Import losses - assuming these files exist in Repository-1 as indicated
from label_relaxation import lr_torch, lr_pairwise, lr_beta, rda_ce
from baseline_loss_functions import GCELoss, NCELoss, AUELoss, EvidenceSmoothingLoss, AGCELoss

def _row_minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min, _ = x.min(dim=1, keepdim=True)
    x_max, _ = x.max(dim=1, keepdim=True)
    denom = (x_max - x_min).clamp_min(eps)
    out = (x - x_min) / denom
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def _soft_label_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(targets * logp).sum(dim=1).mean()

class Llama3LBWRanker(torch.nn.Module):
    def __init__(self, device=None, params=None, pos_lambda: float = 0.001,
                 neg_lambda: float = 0.01, alpha: float = 0.001, margin: float = 1):
        super(Llama3LBWRanker, self).__init__()
        self.params = params
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_name = "meta-llama/Llama-3.2-1B"
        print(f"Loading Llama3 LBW (Look Both Ways) model from {model_name}...")
        
        # 1. Load Config & Init Model
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = LlamaModelLBW.from_pretrained(model_name, config=self.config)
        self.model.to(self.device)
        
        # 2. Tokenizer (Right Padding for Encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def mean_pooling(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        # Weighted mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        return torch.sum(last_hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # Remove token_type_ids if present (Llama doesn't use them)
        batch.pop('token_type_ids', None)
        outputs = self.model(**batch)
        return outputs

    def encode_context(self, batch):
        return self.encode_candidate(batch)

    def encode_candidate(self, batch):
        outputs = self.encode(batch)
        embeddings = self.mean_pooling(outputs.last_hidden_state, batch['attention_mask'])
        return embeddings.detach().cpu()

    # --- Loss and Forward Logic Ported from E5.py ---

    def get_reg(self, inputs, targets):
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

    def loss_gls(self, logits, labels):
        print("Applying gls ----------------------")
        smooth_rate = self.params['label_smoothness']
        confidence = 1. - smooth_rate
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smooth_rate * smooth_loss
        return loss.mean()

    def get_diff(self, scores):
        max_values = scores.max(dim=1).values.unsqueeze(dim=1).repeat(1, scores.shape[1])
        diff = max_values - scores
        return diff

    def mbls_forward(self, scores, target):
        alpha = 0.1
        margin = 10
        loss_ce = F.cross_entropy(scores, target, reduction='mean')
        diff = self.get_diff(scores)
        loss_margin = F.relu(diff - margin).mean()
        loss = loss_ce + alpha * loss_margin
        return loss, scores

    def acls_forward(self, scores, targets):
        loss_ce = F.cross_entropy(scores, targets, reduction="mean")
        loss_reg = self.get_reg(scores, targets)
        loss = loss_ce + self.alpha * loss_reg
        return loss, scores

    def wsls_forward(self, scores, target):
        print("Applying weakly supervised loss...")
        B, C = scores.shape
        row_idx = torch.arange(B, device=scores.device)
        eps_wsls = float(self.params.get('wsls_eps', 0.2))
        
        # Check epoch for two-stage
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

    def per_example_calibration_error(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        correctness = predictions.eq(labels).float()
        return torch.abs(confidences - correctness)

    def compute_recall_at_k(self, scores, k=5):
        topk = torch.topk(scores, k=k, dim=1).indices
        targets = torch.arange(scores.size(0)).unsqueeze(1).to(scores.device)
        match = (topk == targets).any(dim=1).float()
        return match.mean().item()

    def forward_diag(self, input_tokens):
        # Read epoch
        try:
            with open("epoch.txt", "r") as f:
                epoch = int(f.read().strip())
        except:
            epoch = 0

        # Encode
        outputs = self.encode(input_tokens)
        embeddings = self.mean_pooling(outputs.last_hidden_state, input_tokens['attention_mask'])
        
        # Split Context (Query) / Document
        # Assuming batch is 50/50 split as per standard Repo-1 logic
        mid = embeddings.size(0) // 2
        context_embed, document_embed = torch.split(embeddings, mid)
        
        # Compute Scores
        scores = (context_embed @ document_embed.T)
        bs = scores.size(0)
        target = torch.arange(bs, device=self.device)

        # --- Loss Selection Logic ---

        if self.params.get('label_relaxation_pairwise') == 'yes':
            loss_fn = lr_pairwise.LabelRelaxationPairwiseLoss(
                alpha=float(self.params['relaxation_param']),
                dim=-1, logits_provided=True, one_hot_encode_trgts=True, num_classes=bs
            )
            loss = loss_fn(scores, target)
            print('relaxation parameter applied:', self.params['relaxation_param'])
            return loss, scores

        if self.params.get('ambiguation_loss') == 'yes':
            loss_fn = rda_ce.BetaCompleteAmbiguationPairwiseLoss(
                alpha=0.1, beta=0.2, num_classes=bs, adaptive_beta=True,
                epochs=self.params['num_epochs'], adaptive_start_beta=0.5,
                adaptive_end_beta=0.1, adaptive_type="linear"
            )
            loss = loss_fn(scores, target, epoch=epoch+1)
            return loss, scores

        if self.params.get('gce_loss') == 'yes':
            loss_fn = GCELoss.GCELoss(num_classes=bs)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('nce_loss') == 'yes':
            loss_fn = NCELoss.NCELoss(num_classes=bs)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('aue_loss') == 'yes':
            loss_fn = AUELoss.AUELoss(num_classes=bs)
            loss = loss_fn(scores, target)
            return loss, scores

        if self.params.get('mbls') == 'yes':
            return self.mbls_forward(scores, target)

        if self.params.get('acls') == 'yes':
            return self.acls_forward(scores, target)
            
        if self.params.get('wsls') == 'yes':
            return self.wsls_forward(scores, target)

        if self.params.get('adaptive_epoch_smoothing') == 'yes':
            if epoch > self.params.get('epoch_bound', 0):
                self.params['label_smoothness'] = self.params['label_smoothness'] - 1.0

        if self.params.get('label_smoothness', 0.0) < 0.0:
            loss = self.loss_gls(scores, target)
        else:
            loss = F.cross_entropy(scores, target, reduction="mean")

        return loss, scores

    def forward(self, doc_input, context_len=None, target=None):
        """
        Main entry point for Trainer.
        Trainer.py calls: loss, logits = self.model(context_input, candidate_input, labels)
        But wait, Repository-1 TrainerE5 calls: self.model(token_input, len(batch[0]), labels)
        
        We need to support the signature used in Trainer.py.
        """
        # If input is just one dictionary (token_input) as planned in TrainerLlama3LBW
        if isinstance(doc_input, dict):
            # This matches the call signature in our new TrainerLlama3LBW:
            # self.model(token_input, original_query_count, labels)
            # where doc_input=token_input, context_len=query_count, target=labels
            
            # Encode everything
            outputs = self.encode(doc_input)
            embeddings = self.mean_pooling(outputs.last_hidden_state, doc_input['attention_mask'])
            
            # Split
            if context_len is None:
                # Fallback or inference
                return embeddings
                
            query_embeddings = embeddings[:context_len]
            doc_embeddings = embeddings[context_len:]
            
            # Scores
            scores = (query_embeddings @ doc_embeddings.T)
            
            if target is None:
                bs = scores.size(0)
                target = torch.arange(bs, device=self.device)
                
            # Reuse logic from forward_diag by calling it with pre-calculated scores?
            # E5.py actually repeats the logic. We will check the params again.
            
            # --- Logic from forward_diag duplicated here for the Trainer signature ---
            try:
                with open("epoch.txt", "r") as f:
                    epoch = int(f.read().strip())
            except:
                epoch = 0
                
            batch_size, num_classes = scores.size()

            if self.params.get('label_relaxation') == 'yes':
                loss_fn = lr_torch.LabelRelaxationLoss(
                    alpha=float(self.params['relaxation_param']),
                    dim=-1, logits_provided=True, one_hot_encode_trgts=True, num_classes=num_classes
                )
                loss = loss_fn(scores, target)
                return loss, scores
                
            if self.params.get('ebls') == 'yes':
                 loss, _ = EvidenceSmoothingLoss.ebls_loss_rows(
                    scores=scores, cand_embs=doc_embeddings.detach(),
                    target=target, eps=0.1, k=5, lam=0.5, metric='cosine', detach_evidence=True
                )
                 return loss, scores

            if self.params.get('label_smoothness', 0.0) < 0.0:
                loss = self.loss_gls(scores, target)
            else:
                loss = F.cross_entropy(scores, target, reduction="mean", 
                                     label_smoothing=float(self.params.get('label_smoothness', 0.0)))
                                     
            return loss, scores

        else:
            # Fallback if called differently (e.g. from Evaluator)
            return self.forward_diag(doc_input)
