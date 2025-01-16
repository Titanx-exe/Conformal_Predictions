import pickle

import torch
from tqdm import tqdm, trange

class Biencoder_Collator():
    def __init__(self,tokenizer,args,queries,device="cpu"):
        self.tokenizer = tokenizer
        self.args = args
        self.device=device
        self.queries=queries
        self.documents=None

    '''def process_sample(self, query, documents=None, label=None):
        question_tokens = self.tokenizer.tokenize("[CLS]"+self.queries[query]["text"])
        question_ids = self.tokenizer.convert_tokens_to_ids(question_tokens)
        if len(question_ids) > self.args["max_context_length"]:
            cans = question_ids[0:self.args["max_context_length"] - 1]
            cans.append(question_ids[len(question_ids) - 1])
            question_ids = cans
        padding = [0] * (self.args["max_context_length"] - len(question_ids))
        question_ids += padding

        if documents is not None:
            candidate_tokens = []
            for cand in documents:
                cand_tokens = self.tokenizer.tokenize("[CLS]title: "+self.documents[cand][1]+ "[SEP] context: "+self.documents[cand][2])
                cand_ids = self.tokenizer.convert_tokens_to_ids(cand_tokens)
                if len(cand_ids) > self.args["max_cand_length"]:
                    cans = cand_ids[0:self.args["max_cand_length"] - 1]
                    cans.append(cand_ids[len(cand_ids) - 1])
                    cand_ids = cans
                padding = [0] * (self.args["max_cand_length"] - len(cand_ids))
                cand_ids += padding
                candidate_tokens.append(cand_ids)
            candidates = candidate_tokens
        return question_ids, candidates, label
    '''
    def collate_entities(self,batch):
        #candidates = ["[CLS]title: "+self.documents[cand][1]+ "[SEP] context: "+self.documents[cand][2] for cand in batch]
        candidates = [self.documents[cand] for cand in
                      batch]
        batch = self.tokenizer(candidates, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return batch.to(self.device)


    def collate_batch_train(self,batch):
        #question_input = []
        candidate_batch=self.collate_entities([el[1]for el in batch])
        question_batch=self.collate_context(el[0]for el in batch)

        return {"context_input": question_batch,
                "candidate_input": candidate_batch}

    def collate_batch_eval(self,batch):
        candidate_batch = self.collate_entities([el[1] for el in batch])
        question_batch = self.collate_context(el[0] for el in batch)
        labels=[1 for el in batch[0][0]]
        return {"context_input": question_batch,
                "candidate_input": candidate_batch,
                "labels":labels}
    def collate_context(self,batch):
        questions=[self.queries[question]["text"] for question in batch]
        batch = self.tokenizer(questions, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return batch.to(self.device)

    '''def collate_entities(self,batch):
        candidate_tokens = []
        for cand in batch:
            cand_tokens = self.tokenizer.tokenize("[CLS]"+self.documents[cand])
            cand_ids = self.tokenizer.convert_tokens_to_ids(cand_tokens)
            if len(cand_ids) > self.args["max_cand_length"]:
                cans = cand_ids[0:self.args["max_cand_length"] - 1]
                cans.append(cand_ids[len(cand_ids) - 1])
                cand_ids = cans
            padding = [0] * (self.args["max_cand_length"] - len(cand_ids))
            cand_ids += padding
            #
            candidate_tokens.append(cand_ids)
        candidate_tokens
        return torch.tensor(candidate_tokens, device=self.device)

    def collate_batch_train(self,batch):
        question_input = []
        candidate_input = []
        for sample in batch:
            qt,ct,_=self.process_sample(sample[1],[sample[0]])
            question_input.append(qt)
            candidate_input.append(ct[0])
        return {"context_input": torch.tensor(question_input, device=self.device),
                "candidate_input": torch.tensor(candidate_input, device=self.device)}

    def collate_batch_eval(self,batch):
        question_input = []
        candidate_input = []
        for sample in batch:
            qt, ct,_ = self.process_sample(sample[0], sample[1])
            question_input.append(qt)
            candidate_input.append(ct)
        labels=[1 for el in batch[0][0]]
        return {"context_input": torch.tensor(question_input, device=self.device),
                "candidate_input": torch.tensor(candidate_input, device=self.device),
                "labels":labels}

    def collate_context(self,batch):
        question_input = []
        for sample in batch:
            qt, _,_ = self.process_sample(sample)
            question_input.append(qt)
        return torch.tensor(question_input, device=self.device)''
'''
class E5collator:
    def __init__(self,tokenizer,device,queries):
        self.tokenizer=tokenizer
        self.queries = queries
        self.documents = None
        self.device=device
    def collate(self,batch, is_passage):
        if is_passage:
            repr=["passage: "+ self.documents[cand] for cand in
                      batch]
        else:
            repr=["query: "+self.queries[question]["text"] for question in batch]
        return repr
    def collate_entities(self,batch):
        repr=["passage: "+ self.documents[cand] for cand in
                      batch]
        batch=self.tokenizer(repr, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return batch.to(self.device)

    def collate_context(self,batch):
        repr = ["query: " + self.queries[question]["text"] for question in batch]
        batch = self.tokenizer(repr, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return batch.to(self.device)