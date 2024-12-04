import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class E5Ranker(torch.nn.Module):
    def __init__(self):
        super(E5Ranker, self).__init__()
        self.model = AutoModel.from_pretrained('intfloat/e5-base-v2')
        #self.loss_fn = InfoNCE()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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

    def forward(self,input):
        #model_input=torch.cat((context_input,document_input),0)
        outputs = self.encode(input)
        embeddings = self.average_pool(outputs.last_hidden_state, input['attention_mask'])
        context_embed,document_embed=torch.split(embeddings,int(embeddings.size(0)/2))
        scores = (context_embed @ document_embed.T)
        bs = scores.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(self.device)
        loss = F.cross_entropy(scores, target, reduction="mean")
        return loss,scores

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