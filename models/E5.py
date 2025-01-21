from audioop import cross

import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class E5Ranker(torch.nn.Module):
    def __init__(self,device=None):
        super(E5Ranker, self).__init__()
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
    #label_smoothed forward
    '''
    def forward(self, input, smoothing_factor=-0.1):
        """
        Forward pass with label smoothing.
        :param input: Tokenized input data.
        :param smoothing_factor: Float, label smoothing factor (0 disables smoothing).
        :return: Loss and scores.
        """
        # Encode the input and compute embeddings
        outputs = self.encode(input)
        embeddings = self.average_pool(outputs.last_hidden_state, input['attention_mask'])
        context_embed, document_embed = torch.split(embeddings, int(embeddings.size(0) / 2))

        # Compute similarity scores
        scores = context_embed @ document_embed.T
        bs = scores.size(0)

        # Generate smoothed labels
        target = torch.arange(bs).to(self.device)  # True labels

        smoothed_labels = torch.full((bs, bs), smoothing_factor / (bs - 1), device=self.device)
        smoothed_labels.scatter_(1, target.unsqueeze(1), 1.0 - smoothing_factor)

        # Compute loss using smoothed labels
        log_probs = F.log_softmax(scores, dim=-1)
        #loss = -(smoothed_labels * log_probs).sum(dim=-1).mean()
        loss = F.cross_entropy(log_probs, smoothed_labels, reduction="mean")
        return loss, scores
    
    #boundary smoothing forward
    def forward(self, input, smoothing_factor=0.1, top_k=2, threshold=None):
        """
        Forward pass with score-based boundary smoothing.
        :param input: Tokenized input data.
        :param smoothing_factor: Float, smoothing factor (0 disables smoothing).
        :param top_k: Integer, number of top-scoring neighbors to include.
        :param threshold: Float, optional score threshold for selecting neighbors.
        :return: Loss and scores.
        """
        # Encode the input and compute embeddings
        outputs = self.encode(input)
        embeddings = self.average_pool(outputs.last_hidden_state, input['attention_mask'])
        context_embed, document_embed = torch.split(embeddings, int(embeddings.size(0) / 2))

        # Compute similarity scores
        scores = context_embed @ document_embed.T  # Shape: [batch_size, batch_size]
        bs = scores.size(0)

        # Generate score-based smoothed labels
        target = torch.arange(bs).to(self.device)  # True labels (diagonal indices)
        smoothed_labels = torch.zeros((bs, bs), device=self.device)

        for i in range(bs):
            # Correct class probability
            smoothed_labels[i, target[i]] = 1.0 - smoothing_factor

            # Sort scores to determine neighbors
            sorted_indices = torch.argsort(scores[i], descending=True)

            if threshold is not None:
                # Use threshold-based neighbor selection
                neighbors = [j for j in sorted_indices if scores[i, j] >= scores[i, target[i]] - threshold]
            else:
                # Use top-k neighbor selection
                neighbors = sorted_indices[:top_k + 1].tolist()  # Top-k neighbors (including the correct class)
                if target[i].item() in neighbors:
                    neighbors.remove(target[i].item())  # Exclude the correct class

            # Assign probabilities to neighbors
            for neighbor in neighbors:
                smoothed_labels[i, neighbor] += smoothing_factor / len(neighbors)

            # Normalize smoothed labels for numerical stability
        smoothed_labels = torch.clamp(smoothed_labels, min=1e-9, max=1.0)


        # Compute loss using boundary-smoothed labels
        log_probs = F.log_softmax(scores, dim=-1)
        loss = -(smoothed_labels * log_probs).sum(dim=-1).mean()

        return loss, scores

    '''
    def forward(self, input, smoothing_factor=-5.0, smoothing_size=1):
        """
        Forward pass with boundary smoothing.
        :param input: Tokenized input data.
        :param smoothing_factor: Float, smoothing factor (0 disables smoothing).
        :param smoothing_size: Integer, the number of neighbors (on each side) to smooth over.
        :return: Loss and scores.
        """
        # Encode the input and compute embeddings
        outputs = self.encode(input)
        embeddings = self.average_pool(outputs.last_hidden_state, input['attention_mask'])
        context_embed, document_embed = torch.split(embeddings, int(embeddings.size(0) / 2))

        # Compute similarity scores
        scores = context_embed @ document_embed.T  # Shape: [batch_size, batch_size]
        bs = scores.size(0)

        # Generate boundary-smoothed labels
        target = torch.arange(bs).to(self.device)  # True labels (diagonal indices)
        smoothed_labels = torch.zeros((bs, bs), device=self.device)

        for i in range(bs):
            smoothed_labels[i, target[i]] = 1.0 - smoothing_factor  # Assign majority probability to the correct label
        
            # Assign probabilities to neighboring spans/entities
            for offset in range(1, smoothing_size + 1):
                if i - offset >= 0:  # Left neighbor
                    smoothed_labels[i, i - offset] += smoothing_factor / (2 * smoothing_size)
                if i + offset < bs:  # Right neighbor
                    smoothed_labels[i, i + offset] += smoothing_factor / (2 * smoothing_size)

        # Normalize smoothed labels for numerical stability
        smoothed_labels = torch.clamp(smoothed_labels, min=1e-9, max=1.0)

        # Compute loss using boundary-smoothed labels
        log_probs = F.log_softmax(scores, dim=-1)
        loss = -(smoothed_labels * log_probs).sum(dim=-1).mean()

        return loss, scores
    
    '''
    #Ordinary forward
    def forward(self,input):
        #model_input=torch.cat((context_input,document_input),0)
        outputs = self.encode(input)
        embeddings = self.average_pool(outputs.last_hidden_state, input['attention_mask'])
        context_embed,document_embed=torch.split(embeddings,int(embeddings.size(0)/2))
        scores = (context_embed @ document_embed.T)
        bs = scores.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(self.device)

        loss = F.cross_entropy(scores, target, reduction="mean", label_smoothing=-0.1)
        return loss,scores
    '''
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