import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from label_relaxation import lr_torch

class E5Ranker(torch.nn.Module):
    def __init__(self,device=None, params=None,pos_lambda: float = 0.001,
                 neg_lambda: float = 0.01,
                 alpha: float = 0.001,    
                 margin: float = 1):
        super(E5Ranker, self).__init__()
        self.params=params
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin
        self.model = AutoModel.from_pretrained('intfloat/e5-base-v2')
        #self.loss_fn = InfoNCE()
        if device==None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device=device
    def average_pool(self,last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    def encode(self,batch):
        return self.model(**batch)

    def encode_context(self,batch):
        return self.encode_candidate(batch)
    def encode_candidate(self,batch):
        outputs = self.encode(batch)
        embeddings = self.average_pool(outputs.last_hidden_state, batch['attention_mask'])
        return embeddings.cpu().detach()

    def forward_diag(self, input):
        EPOCH_FILE = "epoch.txt"
        with open(EPOCH_FILE, "r") as f:
            epoch = int(f.read().strip())
        #model_input=torch.cat((context_input,document_input),0)
        outputs = self.encode(input)
        embeddings = self.average_pool(outputs.last_hidden_state, input['attention_mask'])
        context_embed,document_embed=torch.split(embeddings,int(embeddings.size(0)/2))
        scores = (context_embed @ document_embed.T)
        bs = scores.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(self.device)
        #print("???????????????????????????????? target is_--------------------------")
        #print(target)
        #print("##################################### score is_--------------------------")
        #print(scores)

        #print("************************************************", self.params['mbls'])
        if self.params['mbls'] == 'yes':
            return self.mbls_forward(scores, target)
        if self.params['acls'] == 'yes':
            return self.acls_forward(scores, target)
        
        if self.params['adaptive_epoch_smoothing'] == 'yes':
            if epoch > self.params['epoch_bound']:
                self.params['label_smoothness'] = self.params['label_smoothness'] - 1.0
                
        if self.params['label_smoothness'] < 0.0:
            loss = self.loss_gls(scores, target)
        else:
            loss = F.cross_entropy(scores, target, reduction="mean")
        
        #loss = F.cross_entropy(scores, target, reduction="mean")
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

        

        pos = indicator * pos_dist_margin ** 2
        neg = (1.0 - indicator) * (neg_dist_margin ** 2)

        reg = self.pos_lambda * (pos.sum() / num_pos) + self.neg_lambda * (neg.sum() / num_neg)

        print(f"Reg Pos Part: {(pos.sum() / num_pos).item():.4f}")
        print(f"Reg Neg Part: {(neg.sum() / num_neg).item():.4f}")

        
        return reg


    def acls_forward(self, scores, targets, alpha=0.1):
        if scores.dim() > 2:
            scores = scores.view(scores.size(0), scores.size(1), -1)  # N,C,H,W => N,C,H*W
            scores = scores.transpose(1, 2)    # N,C,H*W => N,H*W,C
            scores = scores.contiguous().view(-1, scores.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        loss_ce = F.cross_entropy(scores, targets, reduction="mean")
        #print("????????????????????????????????cross entropy loss is_--------------------------")
        #print(loss_ce)

        print("*************************************** ACLS applied ********************************")
        loss_reg = self.get_reg(scores, targets)
        loss = loss_ce + self.alpha * loss_reg

        #print("#################################### loss_reg is ++++++++++++++++++++++++++++")
        #print(loss_reg)
        
        return loss, scores

    #negative label smoothing
    def nls_forward(self, scores, target=None):
        """
        Forward pass with negative label smoothing.
        :param scores: Logits tensor of shape (batch_size, num_classes).
        :param target: Optional tensor containing correct class indices.
        :return: Loss and scores.
        """
        smoothing_factor = self.params['label_smoothness']
        bs, num_classes = scores.shape  # Get batch size and number of classes

        # If target is None, assign default sequential target labels
        if target is None:
            target = torch.arange(bs, device=self.device)

        # Ensure target indices are within the valid range
        if target.max() >= num_classes or target.min() < 0:
            raise ValueError(f"Target indices out of range. Target max: {target.max()}, num_classes: {num_classes}")

        # Initialize smoothed label tensor with uniform distribution
        smoothed_labels = torch.full((bs, num_classes), smoothing_factor / (num_classes - 1), device=self.device)

        # Assign (1 - smoothing_factor) to the correct target indices
        smoothed_labels.scatter_(1, target.unsqueeze(1), 1.0 - smoothing_factor)

        # Apply Negative Label Smoothing (NLS) transformation if required
        if smoothing_factor < 0:
            neg_factor = -smoothing_factor
            smoothed_labels = (1 + neg_factor) * smoothed_labels - (neg_factor / (num_classes - 1))
            smoothed_labels = smoothed_labels / smoothed_labels.sum(dim=1, keepdim=True)  # Normalize

        #print("-------------------Smoothed Labels--------------------------")
        #print(smoothed_labels)
    
        # Compute log probabilities
        log_probs = F.log_softmax(scores, dim=-1)
        #print("-------------------Log Probabilities--------------------------")
        #print(log_probs)

        # Compute KL divergence loss
        loss = F.kl_div(log_probs, smoothed_labels, reduction="batchmean")

        return loss, scores

    def loss_gls(self, logits, labels):
        # logits: model prediction logits before the soft-max, with size [batch_size, classes]
        # labels: the (noisy) labels for evaluation, with size [batch_size]
        # smooth_rate: could go either positive or negative,
        # smooth_rate candidates we adopted in the paper: [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0].
        #print("labels are############################", labels)
        print("Applying gls ----------------------")
        smooth_rate = self.params['label_smoothness']
        confidence = 1. - smooth_rate
        logprobs = F.log_softmax(logits, dim=-1)
        #print("log probabilities are++++++++++++++++++++++++++++", logprobs)
        #print("labels are***************************************", labels)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        #print("labels are############################", nll_loss)
        nll_loss = nll_loss.squeeze(1)
        #print("nll_loss is $$$$$$$$$$$$$$$$$$$$$$$$$$$$", nll_loss)
        smooth_loss = -logprobs.mean(dim=-1)
        #print("smoothed loss is??????????????????????????????", smooth_loss)
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
            scores = scores.view(scores.size(0), scores.size(1), -1)  # N,C,H,W => N,C,H*W
            scores = scores.transpose(1, 2)    # N,C,H*W => N,H*W,C
            scores = scores.contiguous().view(-1, scores.size(2))   # N,H*W,C => N*H*W,C
            scores = scores.view(-1)
            
        loss_ce = F.cross_entropy(scores, target, reduction='mean')
        
        #print("scores is ++++++++++++++++++++++++++++++++++", scores)
        # get logit distance
        diff = self.get_diff(scores)
        #print("differences are ...........................", diff)
        # linear penalty where logit distances are larger than the margin
        loss_margin = F.relu(diff-margin).mean()
        loss = loss_ce + alpha * loss_margin
        #print("loss ce is#################################", loss_ce)
        #print("loss is%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", loss)

        return loss, scores
    
    def forward(self,doc_input,context_len=None,target=None):
        EPOCH_FILE = "epoch.txt"
        with open(EPOCH_FILE, "r") as f:
            epoch = int(f.read().strip())

        if context_len is None:
            return self.forward_diag(doc_input)

        output = self.encode(doc_input)
        embeddings = self.average_pool(output.last_hidden_state, doc_input['attention_mask'])
        context_embeddings, candidate_embeddings = torch.split(embeddings, [context_len,embeddings.size(0)-context_len])
        scores = (context_embeddings @ candidate_embeddings.T)
        
        if target is None:
            bs = scores.size(0)
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
        #print("????????????????????????????????target is_--------------------------")
        #print(target)

        if self.params['label_relaxation'] == 'yes':
            batch_size, num_classes = scores.size()
            loss_fn = lr_torch.LabelRelaxationLoss(
                alpha=float(self.params['relaxation_param']),
                dim=-1,
                logits_provided=True,
                one_hot_encode_trgts=True,
                num_classes= num_classes # must match your actual number of classes
            )
            loss = loss_fn(scores, target)
            #print("------------loss is ++++++++++++++++++++")
            #print(loss)

            #print("scores ................................")
            #print(scores)
            return loss, scores

        if self.params['mbls'] == 'yes':
            return self.mbls_forward(scores, target)
        if self.params['acls'] == 'yes':
            return self.acls_forward(scores, target)    

        if self.params['adaptive_epoch'] == 'yes':
            if epoch > self.params['epoch_bound']:
                self.params['label_smoothness'] = self.params['label_smoothness'] - 1.0
        
       
        if self.params['label_smoothness'] < 0.0:
            #loss, scores = self.nls_forward(scores, target)
            loss = self.loss_gls(scores, target)
        else:
            print("Applying cross entropy with label_smoothness ######################", self.params['label_smoothness'])
            loss = F.cross_entropy(scores, target, reduction="mean", label_smoothing=float(self.params['label_smoothness']))
        #else:
            #print("Applying cross entropy ######################")
        #    loss = F.cross_entropy(scores, target, reduction="mean")
            
        return loss, scores


# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
'''
context_texts = ['query: how much protein should a female eat',
               'query: summit define',
               ]

document_texts = [
                "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
               "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day."
               ]
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
model = E5Ranker()

context_texts.extend(document_texts)
# Tokenize the input texts

documents = tokenizer(context_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(documents)
'''

# normalize embeddings
#embeddings = F.normalize(embeddings, p=2, dim=1)
#r=embeddings[:2] @ embeddings[2:].T
#scores = (embeddings[:2] @ embeddings[2:].T) * 100
#print(scores.tolist())