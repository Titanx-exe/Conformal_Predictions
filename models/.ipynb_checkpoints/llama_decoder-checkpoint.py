# models/llama_decoder.py

import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoConfig

# Import custom LlamaModel from Decoder folder
from models.Decoder.modeling_llama import LlamaModel

# Import loss functions (same as E5.py)
from label_relaxation import lr_torch, lr_pairwise, lr_beta, rda_ce
from baseline_loss_functions import (
    GCELoss,
    NCELoss,
    AUELoss,
    EvidenceSmoothingLoss,
    AGCELoss,
)

# LoRA adapter support
from peft import PeftModel


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


class LlamaDecoderRanker(torch.nn.Module):
    def __init__(
        self,
        device=None,
        params=None,
        pos_lambda: float = 0.001,
        neg_lambda: float = 0.01,
        alpha: float = 0.001,
        margin: float = 1,
    ):
        super().__init__()

        self.params = params if params else {}
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # LoRA adapter configuration
        lora_adapter_path = params.get("lora_adapter_path", None)
        lora_epoch = params.get("lora_epoch", "final")

        # Model configuration
        model_id = params.get("model_id", "meta-llama/Llama-3.2-3B")

        # 1. Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.tokenizer.padding_side = "right"  # CRITICAL: Right padding for LBW
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load config and set LBW parameters
        self.config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config._attn_implementation = "eager"  # or "flash_attention_2"

        # LBW Configuration
        # BIDIR: removes causal mask -> all tokens attend to all tokens (like BERT)
        # BACK: reversed causal mask -> each token attends to itself and SUBSEQUENT tokens
        self.config.architecture = params.get("lbw_architecture", "INPLACE")
        self.config.num_bidir_layers = params.get(
            "num_bidir_layers", 10
        )  # BIDIR layers
        self.config.num_unsink_layers = params.get(
            "num_unsink_layers", 0
        )  # BACK layers
        self.config.mask_type = params.get("mask_type", "MASK0")  # 'MASK0' or 'BACK'

        # 3. Load model with custom config
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = LlamaModel.from_pretrained(
            model_id,
            config=self.config,
            torch_dtype=dtype,
            local_files_only=True,
        )
        self.model.to(self.device)

        # Load LoRA adapter if specified
        if lora_adapter_path is not None:
            full_adapter_path = os.path.join(lora_adapter_path, lora_epoch)
            print(f" Loading LoRA adapter from: {full_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, full_adapter_path, local_files_only=True
            )
            self.model = self.model.merge_and_unload()
            self.model.to(self.device)
        print(f" LoRA adapter loaded and merged successfully! (epoch: {lora_epoch})")

        print(f" LlamaDecoderRanker initialized on {self.device}")
        print(f" Architecture: {self.config.architecture}")
        print(
            f"  BIDIR layers: {self.config.num_bidir_layers} (bidirectional attention)"
        )
        print(
            f"  BACK layers: {self.config.num_unsink_layers} (backward attention, mask_type={self.config.mask_type})"
        )

    def mean_pooling(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Mean pooling over non-padding tokens.
        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, batch):
        """
        Forward pass through LlamaModel.
        Remove token_type_ids if present - LLaMA doesn't use them.
        """
        batch = {
            k: v.to(self.device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        batch.pop("token_type_ids", None)
        return self.model(**batch)

    def encode_context(self, batch):
        """Encode queries."""
        return self.encode_candidate(batch)

    def encode_candidate(self, batch):
        """Encode documents and return embeddings on CPU."""
        outputs = self.encode(batch)
        # embeddings = self.mean_pooling(outputs.last_hidden_state, batch['attention_mask'])
        mask = batch["attention_mask"].to(self.device)
        embeddings = self.mean_pooling(outputs.last_hidden_state, mask)
        return embeddings.cpu().detach()

    def forward_diag(self, input):
        EPOCH_FILE = "epoch.txt"
        with open(EPOCH_FILE, "r") as f:
            epoch = int(f.read().strip())
        # model_input=torch.cat((context_input,document_input),0)
        outputs = self.encode(input)
        embeddings = self.mean_pooling(
            outputs.last_hidden_state, input["attention_mask"]
        )
        context_embed, document_embed = torch.split(
            embeddings, int(embeddings.size(0) / 2)
        )
        scores = context_embed @ document_embed.T
        bs = scores.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(self.device)
        # print("???????????????????????????????? target is_--------------------------")
        # print(target)
        # print("##################################### score is_--------------------------")
        # print(scores)

        # print("************************************************", self.params['mbls'])

        if self.params["label_relaxation_pairwise"] == "yes":
            batch_size, num_classes = scores.size()
            loss_fn = lr_pairwise.LabelRelaxationPairwiseLoss(
                alpha=float(self.params["relaxation_param"]),
                dim=-1,
                logits_provided=True,
                one_hot_encode_trgts=True,
                num_classes=num_classes,  # must match your actual number of classes
            )
            loss = loss_fn(scores, target)
            print(
                "relaxation parameter applied------------------:",
                self.params["relaxation_param"],
            )
            # print("------------loss is ++++++++++++++++++++")
            # print(loss)

            # print("scores ................................")
            # print(scores)
            return loss, scores

        if self.params["ambiguation_loss"] == "yes":
            batch_size, num_classes = scores.size()
            num_epochs = self.params["num_epochs"]
            loss_fn = rda_ce.BetaCompleteAmbiguationPairwiseLoss(
                alpha=0.1,
                beta=0.2,
                num_classes=num_classes,
                adaptive_beta=True,
                epochs=num_epochs,
                adaptive_start_beta=0.5,
                adaptive_end_beta=0.1,
                adaptive_type="linear",
            )
            loss = loss_fn(scores, target, epoch=epoch + 1)
            # print("------------loss is ++++++++++++++++++++")
            # print(loss)
            # print("Epochs ................................")
            # print(epoch)
            return loss, scores

        if self.params["gce_loss"] == "yes":
            batch_size, num_classes = scores.size()
            loss_fn = GCELoss.GCELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            print(
                "*****************************************************************************"
            )
            print("------------loss is ++++++++++++++++++++")
            print(loss)
            # print("Epochs ................................")
            # print(epoch)
            return loss, scores
        if self.params["nce_loss"] == "yes":
            batch_size, num_classes = scores.size()
            loss_fn = NCELoss.NCELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            print(
                "*****************************************************************************"
            )
            print("------------loss is ++++++++++++++++++++")
            print(loss)
            # print("Epochs ................................")
            # print(epoch)
            return loss, scores

        if self.params["aue_loss"] == "yes":
            batch_size, num_classes = scores.size()
            loss_fn = AUELoss.AUELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            # print("------------loss is ++++++++++++++++++++")
            # print(loss)

            # print("Epochs ................................")
            # print(epoch)

            return loss, scores

        if self.params["mbls"] == "yes":
            return self.mbls_forward(scores, target)
        if self.params["acls"] == "yes":
            return self.acls_forward(scores, target)

        if self.params["adaptive_epoch_smoothing"] == "yes":
            if epoch > self.params["epoch_bound"]:
                self.params["label_smoothness"] = self.params["label_smoothness"] - 1.0

        if self.params["label_smoothness"] < 0.0:
            loss = self.loss_gls(scores, target)
        else:
            loss = F.cross_entropy(scores, target, reduction="mean")

        # loss = F.cross_entropy(scores, target, reduction="mean")
        return loss, scores

    def wsls_forward(
        self,
        scores: torch.Tensor,  # [B, C]
        target: torch.Tensor,  # [B]
        ns_scores: torch.Tensor | None = None,  # optional, ignored here
    ):
        """
        Weakly Supervised Label Smoothing using *off-diagonal* scores as weak labels.
        Diagonal indices (target) are the true docs; all other columns are negatives.
        """
        print("Applying weakly supervised loss..................................")
        B, C = scores.shape
        device = scores.device
        row_idx = torch.arange(B, device=device)

        # epsilon schedule (optional two-stage)
        eps_wsls = float(self.params.get("wsls_eps", 0.2))
        try:
            with open("epoch.txt", "r") as f:
                cur_epoch = int(f.read().strip())
        except Exception:
            cur_epoch = 0
        if self.params.get("wsls_two_stage", "no") == "yes":
            switch_ep = int(self.params.get("wsls_switch_epoch", max(cur_epoch + 1, 1)))
            if cur_epoch >= switch_ep:
                eps_wsls = 0.0

        # optional temperature to shape negatives
        tau = float(self.params.get("wsls_tau", 1.0))

        # ---- weak distribution from off-diagonals ----
        s = scores.detach()
        if tau != 1.0:
            s = s / tau

        # exclude the diagonal from competing during normalization
        s_neg = s.clone()
        row_min = s_neg.min(dim=1, keepdim=True).values
        s_neg[row_idx, target] = (
            row_min.squeeze(1) - 1.0
        )  # ensure strictly below all negatives

        # min-max normalize row-wise -> [0,1]
        w = _row_minmax_norm(s_neg)

        # set gold index to neutral mass 1/C, then renormalize to a prob. distribution
        w[row_idx, target] = 1.0 / C
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # ---- final soft targets: (1-ε)*onehot + ε*w ----
        one_hot = F.one_hot(target, num_classes=C).float()
        tgt_soft = (1.0 - eps_wsls) * one_hot + eps_wsls * w

        # ---- compute loss ----
        loss = _soft_label_ce(scores, tgt_soft)
        return loss, scores

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

        pos = indicator * pos_dist_margin**2
        neg = (1.0 - indicator) * (neg_dist_margin**2)

        reg = self.pos_lambda * (pos.sum() / num_pos) + self.neg_lambda * (
            neg.sum() / num_neg
        )

        print(f"Reg Pos Part: {(pos.sum() / num_pos).item():.4f}")
        print(f"Reg Neg Part: {(neg.sum() / num_neg).item():.4f}")

        return reg

    def acls_forward(self, scores, targets, alpha=0.1):
        if scores.dim() > 2:
            scores = scores.view(
                scores.size(0), scores.size(1), -1
            )  # N,C,H,W => N,C,H*W
            scores = scores.transpose(1, 2)  # N,C,H*W => N,H*W,C
            scores = scores.contiguous().view(-1, scores.size(2))  # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        loss_ce = F.cross_entropy(scores, targets, reduction="mean")
        # print("????????????????????????????????cross entropy loss is_--------------------------")
        # print(loss_ce)

        print(
            "*************************************** ACLS applied ********************************"
        )
        loss_reg = self.get_reg(scores, targets)
        loss = loss_ce + self.alpha * loss_reg

        # print("#################################### loss_reg is ++++++++++++++++++++++++++++")
        # print(loss_reg)

        return loss, scores

    # negative label smoothing
    def nls_forward(self, scores, target=None):
        """
        Forward pass with negative label smoothing.
        :param scores: Logits tensor of shape (batch_size, num_classes).
        :param target: Optional tensor containing correct class indices.
        :return: Loss and scores.
        """
        smoothing_factor = self.params["label_smoothness"]
        bs, num_classes = scores.shape  # Get batch size and number of classes

        # If target is None, assign default sequential target labels
        if target is None:
            target = torch.arange(bs, device=self.device)

        # Ensure target indices are within the valid range
        if target.max() >= num_classes or target.min() < 0:
            raise ValueError(
                f"Target indices out of range. Target max: {target.max()}, num_classes: {num_classes}"
            )

        # Initialize smoothed label tensor with uniform distribution
        smoothed_labels = torch.full(
            (bs, num_classes), smoothing_factor / (num_classes - 1), device=self.device
        )

        # Assign (1 - smoothing_factor) to the correct target indices
        smoothed_labels.scatter_(1, target.unsqueeze(1), 1.0 - smoothing_factor)

        # Apply Negative Label Smoothing (NLS) transformation if required
        if smoothing_factor < 0:
            neg_factor = -smoothing_factor
            smoothed_labels = (1 + neg_factor) * smoothed_labels - (
                neg_factor / (num_classes - 1)
            )
            smoothed_labels = smoothed_labels / smoothed_labels.sum(
                dim=1, keepdim=True
            )  # Normalize

        # print("-------------------Smoothed Labels--------------------------")
        # print(smoothed_labels)

        # Compute log probabilities
        log_probs = F.log_softmax(scores, dim=-1)
        # print("-------------------Log Probabilities--------------------------")
        # print(log_probs)

        # Compute KL divergence loss
        loss = F.kl_div(log_probs, smoothed_labels, reduction="batchmean")

        return loss, scores

    def loss_gls(self, logits, labels):
        # logits: model prediction logits before the soft-max, with size [batch_size, classes]
        # labels: the (noisy) labels for evaluation, with size [batch_size]
        # smooth_rate: could go either positive or negative,
        # smooth_rate candidates we adopted in the paper: [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0].
        # print("labels are############################", labels)
        print("Applying gls ----------------------")
        smooth_rate = self.params["label_smoothness"]
        confidence = 1.0 - smooth_rate
        logprobs = F.log_softmax(logits, dim=-1)
        # print("log probabilities are++++++++++++++++++++++++++++", logprobs)
        # print("labels are***************************************", labels)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        # print("labels are############################", nll_loss)
        nll_loss = nll_loss.squeeze(1)
        # print("nll_loss is $$$$$$$$$$$$$$$$$$$$$$$$$$$$", nll_loss)
        smooth_loss = -logprobs.mean(dim=-1)
        # print("smoothed loss is??????????????????????????????", smooth_loss)
        loss = confidence * nll_loss + smooth_rate * smooth_loss
        loss_numpy = loss.data.cpu().numpy()
        num_batch = len(loss_numpy)
        return torch.sum(loss) / num_batch

    def get_diff(self, scores):
        max_values = scores.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, scores.shape[1])
        diff = max_values - scores
        return diff

    def mbls_forward(self, scores, target):
        alpha = 0.1
        margin = 10

        if scores.dim() > 2:
            scores = scores.view(
                scores.size(0), scores.size(1), -1
            )  # N,C,H,W => N,C,H*W
            scores = scores.transpose(1, 2)  # N,C,H*W => N,H*W,C
            scores = scores.contiguous().view(-1, scores.size(2))  # N,H*W,C => N*H*W,C
            scores = scores.view(-1)

        loss_ce = F.cross_entropy(scores, target, reduction="mean")

        # print("scores is ++++++++++++++++++++++++++++++++++", scores)
        # get logit distance
        diff = self.get_diff(scores)
        # print("differences are ...........................", diff)
        # linear penalty where logit distances are larger than the margin
        loss_margin = F.relu(diff - margin).mean()
        loss = loss_ce + alpha * loss_margin
        # print("loss ce is#################################", loss_ce)
        # print("loss is%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", loss)

        return loss, scores

    def per_example_calibration_error(self, logits, labels):
        """
        Compute |confidence - correctness| per example.
        logits: [B, C] - model outputs
        labels: [B] - true labels (e.g., diagonal: [0, 1, ..., B-1])
        Returns a tensor of shape [B]
        """
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        correctness = predictions.eq(labels).float()
        return torch.abs(confidences - correctness)

    def compute_recall_at_k(self, scores, k=5):
        """
        scores: [B, B] similarity matrix
        returns: recall@k
        """
        topk = torch.topk(scores, k=k, dim=1).indices  # top-k predictions for each row
        targets = (
            torch.arange(scores.size(0)).unsqueeze(1).to(scores.device)
        )  # correct indices
        match = (topk == targets).any(dim=1).float()  # check if correct is in top-k
        recall_at_k = match.mean().item()
        return recall_at_k

    # ... Copy ALL loss functions from E5.py verbatim ...
    # forward_diag(), forward(), wsls_forward(), acls_forward(), mbls_forward(), etc.

    def forward(self, doc_input, context_len=None, target=None):
        EPOCH_FILE = "epoch.txt"
        with open(EPOCH_FILE, "r") as f:
            epoch = int(f.read().strip())

        if context_len is None:
            return self.forward_diag(doc_input)

        output = self.encode(doc_input)
        embeddings = self.mean_pooling(
            output.last_hidden_state, doc_input["attention_mask"]
        )
        context_embeddings, candidate_embeddings = torch.split(
            embeddings, [context_len, embeddings.size(0) - context_len]
        )
        scores = context_embeddings @ candidate_embeddings.T

        if target is None:
            bs = scores.size(0)
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)

        self.per_example_calibration_error(scores, target)

        # if self.params['label_relaxation'] == 'yes' and epoch > 0 and self.params['smooth2relax'] == 'yes':
        if self.params["label_relaxation"] == "yes":
            batch_size, num_classes = scores.size()
            loss_fn = lr_torch.LabelRelaxationLoss(
                alpha=float(self.params["relaxation_param"]),
                dim=-1,
                logits_provided=True,
                one_hot_encode_trgts=True,
                num_classes=num_classes,  # must match your actual number of classes
            )
            loss = loss_fn(scores, target)
            # print("------------loss is ++++++++++++++++++++")
            # print(loss)
            print(
                "Label relaxation applied with parameter-----------------: "
                + str(self.params["relaxation_param"])
            )
            # print("scores ................................")
            # print(scores)
            return loss, scores

        if self.params["label_relaxation_pairwise"] == "yes":
            batch_size, num_classes = scores.size()
            loss_fn = lr_pairwise.LabelRelaxationPairwiseLoss(
                alpha=float(self.params["relaxation_param"]),
                dim=-1,
                logits_provided=True,
                one_hot_encode_trgts=True,
                num_classes=num_classes,  # must match your actual number of classes
            )
            loss = loss_fn(scores, target)
            print(
                "Label relaxation applied with parameter-----------------: "
                + str(self.params["relaxation_param"])
            )
            # print("------------loss is ++++++++++++++++++++")
            # print(loss)

            # print("scores ................................")
            # print(scores)
            return loss, scores

        if self.params["ambiguation_loss"] == "yes":
            batch_size, num_classes = scores.size()
            num_epochs = self.params["num_epochs"]
            loss_fn = rda_ce.BetaCompleteAmbiguationPairwiseLoss(
                alpha=0.1,
                beta=0.2,
                num_classes=num_classes,
                adaptive_beta=True,
                epochs=num_epochs,
                adaptive_start_beta=0.5,
                adaptive_end_beta=0.1,
                adaptive_type="linear",
            )
            loss = loss_fn(scores, target, epoch=epoch + 1)
            # print("------------loss is ++++++++++++++++++++")
            # print(loss)

            # print("Epochs ................................")
            # print(epoch)

            recall_5 = self.compute_recall_at_k(scores, k=5)

            return loss, scores

        if self.params["gce_loss"] == "yes":
            batch_size, num_classes = scores.size()
            loss_fn = GCELoss.GCELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            # print("------------loss is ++++++++++++++++++++")
            # print(loss)

            # print("Epochs ................................")
            # print(epoch)

            return loss, scores

        if self.params["nce_loss"] == "yes":
            batch_size, num_classes = scores.size()
            loss_fn = NCELoss.NCELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            # print("------------loss is ++++++++++++++++++++")
            # print(loss)

            # print("Epochs ................................")
            # print(epoch)

            return loss, scores

        if self.params["aue_loss"] == "yes":
            batch_size, num_classes = scores.size()
            loss_fn = AUELoss.AUELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            # print("------------loss is ++++++++++++++++++++")
            # print(loss)

            # print("Epochs ................................")
            # print(epoch)

            return loss, scores

        if self.params["agce"] == "yes":
            batch_size, num_classes = scores.size()
            loss_fn = AGCELoss.AGCELoss(num_classes=num_classes)
            loss = loss_fn(scores, target)
            # print("------------loss is ++++++++++++++++++++")
            # print(loss)

            # print("Epochs ................................")
            # print(epoch)

            return loss, scores

        # ----- EBLS with k-reciprocal neighborhoods -----
        if self.params.get("ebls", "no") == "yes":
            eps = 0.1
            k = 5  # no of neighbors
            lam = 0.5
            metric = "cosine"  # 'cosine' or 'dot'

            loss, _ = EvidenceSmoothingLoss.ebls_loss_rows(
                scores=scores,  # [Q, C]
                cand_embs=candidate_embeddings.detach(),  # [C, D]
                target=target,  # [Q] gold column per row (you already have this)
                eps=eps,
                k=k,
                lam=lam,
                metric=metric,
                detach_evidence=True,
            )

            return loss, scores

        if self.params.get("wsls", "no") == "yes":
            return self.wsls_forward(scores, target)

        if self.params["mbls"] == "yes":
            return self.mbls_forward(scores, target)
        if self.params["acls"] == "yes":
            return self.acls_forward(scores, target)

        if self.params["adaptive_epoch"] == "yes":
            if epoch > self.params["epoch_bound"]:
                self.params["label_smoothness"] = self.params["label_smoothness"] - 1.0

        if self.params["label_smoothness"] < 0.0:
            # loss, scores = self.nls_forward(scores, target)
            loss = self.loss_gls(scores, target)
        else:
            if (
                self.params["label_relaxation"] == "yes"
                and self.params["smooth2relax"] == "yes"
            ):
                self.params["label_smoothness"] = self.params["relaxation_param"] - 0.05
            print(
                "Applying cross entropy with label_smoothness ######################",
                self.params["label_smoothness"],
            )
            loss = F.cross_entropy(
                scores,
                target,
                reduction="mean",
                label_smoothing=float(self.params["label_smoothness"]),
            )
        # else:
        # print("Applying cross entropy ######################")
        #    loss = F.cross_entropy(scores, target, reduction="mean")

        return loss, scores
