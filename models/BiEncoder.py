import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

#from blinkRanker.model.ranker_base import BertEncoder, get_model_obj


# from biencoderup.common.optimizer import get_bert_optimizer




def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None,device=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        if device==None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device=device
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            # state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
            state_dict = torch.load(fname, map_location=self.device)
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    # this function is added to compute negative label smoothing
    '''
    def loss_gls(self, logits, labels):
        """
        Computes Generalized Label Smoothing (GLS) loss.
        :param logits: Model logits before softmax, shape [batch_size, num_classes].
        :param labels: True labels, shape [batch_size].
        :return: Smoothed loss.
        """
        batch_size, num_classes = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)

        smoothing_factor = self.params['label_smoothness']
        # Create smoothed label distribution
        confidence = 1.0 - smoothing_factor
        smoothed_labels = torch.full((batch_size, num_classes), smoothing_factor / (num_classes - 1),
                                     device=self.device)
        smoothed_labels.scatter_(1, labels.unsqueeze(1), confidence)

        

        # Apply Negative Label Smoothing (NLS)
        if smoothing_factor < 0:
            neg_factor = -smoothing_factor
            smoothed_labels = (1 + neg_factor) * smoothed_labels - (neg_factor / (num_classes - 1))
            smoothed_labels = smoothed_labels / smoothed_labels.sum(dim=1, keepdim=True)  # Normalize
        #print("performing negative label_smoothing-------------", smoothed_labels)
        return F.kl_div(log_probs, smoothed_labels, reduction="batchmean")

    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    '''
    '''
    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    '''

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):

        batch_size = cand_vecs.size(0)
        candidate_size = cand_vecs.size(1)
        if not random_negs:
            cand_vecs = cand_vecs.view(batch_size * candidate_size, self.params["max_cand_length"])
        # ret=cand_vecs_view.view(10,5,128)
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        #embedding_cands = embedding_cands.view(batch_size, candidate_size, embedding_cands.size(1))
        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_cands = embedding_cands.view(batch_size, candidate_size, embedding_cands.size(1))
            embedding_ctxt = embedding_ctxt.unsqueeze(2)  # batchsize x 1 x embed_size
            # embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_cands, embedding_ctxt)  # batchsize x 1 x 1
            scores = torch.squeeze(scores, dim=2)
            return scores

    # label_input -- negatives provided

    def loss_gls_mod(self, bs, target, scores):
        smoothing_factor = self.params['label_smoothness']
        # Encode the input and compute embeddings

        smoothed_labels = torch.full((bs, bs), smoothing_factor / (bs - 1), device=self.device)
        smoothed_labels.scatter_(1, target.unsqueeze(1), 1.0 - smoothing_factor)

        # Apply Negative Label Smoothing (NLS)
        if smoothing_factor < 0:
            neg_factor = -smoothing_factor
            smoothed_labels = (1 + neg_factor) * smoothed_labels - (neg_factor / (bs - 1))
            # Renormalize to ensure sum = 1
            smoothed_labels = smoothed_labels / smoothed_labels.sum(dim=1, keepdim=True)

        #print("-------------------smoothed labels are:_--------------------------")
        #print(smoothed_labels)
        # Compute loss using smoothed labels
        print(scores)
        # Step 1: Mean-center logits (Row-wise normalization)
        scores = scores - scores.mean(dim=1, keepdim=True)
        # Step 2: Apply temperature scaling
        scores = scores / 0.1
        # Step 3: Clamp extreme values (Only if necessary)
        scores = torch.clamp(scores, min=-50, max=50)
        # Step 4: Compute log-softmax
        log_probs = F.log_softmax(scores, dim=-1)
        print(log_probs)

        loss = F.kl_div(log_probs, smoothed_labels, reduction="batchmean")  # KL-Divergence for soft labels
        #loss = -torch.sum(smoothed_labels * log_probs, dim=-1).mean()


        return loss

    # If label_input is None, train on in-batch negatives
    # new forward function calling negative label_smoothing
    def ls_forward(self, context_input, cand_input, label_input=None):
        """
        Forward pass with Negative Label Smoothing (NLS) support.
        :param context_input: Context embeddings.
        :param cand_input: Candidate embeddings.
        :param label_input: Ground-truth labels (optional).
        :return: Loss and scores.
        """
        flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, flag)
        bs = scores.size(0)

        if label_input is None:

            #target = torch.arange(bs, dtype=torch.long, device=self.device)
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            loss = self.loss_gls_mod(bs, target, scores)
        else:

            loss = self.loss_gls_mod(bs, label_input, scores)

        return loss, scores


    #original forward function

    def forward(self, context_input, cand_input, label_input=None):
        EPOCH_FILE = "epoch.txt"
        with open(EPOCH_FILE, "r") as f:
            epoch = int(f.read().strip())

        if epoch < 10:
            loss, scores = self.ls_forward(context_input, cand_input, label_input)
            #print("----------------Now applying SMOOTHING------------------", epoch)
        else:
            #print("----------------Now applying CROSS ENTROPY------------------", epoch)
            flag = label_input is None
            scores = self.score_candidate(context_input, cand_input, flag)
            bs = scores.size(0)
            if label_input is None:
                target = torch.LongTensor(torch.arange(bs))
                target = target.to(self.device)
                #if epoch > 3:
                loss = F.cross_entropy(scores, target, reduction="mean")
            else:
                # loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
                # TODO: add parameters?
                loss = F.cross_entropy(scores, label_input, reduction="mean")
        return loss, scores

    def predict(self, context_input, cand_input):
        # flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, False)
        return scores


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask

class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result