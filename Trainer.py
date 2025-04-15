import pickle
from optimizers import standard_optimizer
import torch
#from blinkRanker import collator
from models.BiEncoder import BiEncoderRanker
from models.BiEncoderHuggingface import BiEncoderRanker as hf_Ranker
from models.E5 import E5Ranker
import random
from collator import Biencoder_Collator,Biencoder_Collator_Huggingface,E5collator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class TrainerRanker:
    def __init__(self, params, evaluate_after_batch,  device):
        self.grad_acc_steps = params["gradient_accumulation_steps"]
        self.params = params
        self.evaluate_after = evaluate_after_batch
        #self.candidate_size = candidate_size
        self.device = device
        self.model = BiEncoderRanker(params,device=device)
        self.collator = Biencoder_Collator(tokenizer=self.model.tokenizer, args=params,
                                 device=device)

        self.model.to(device)
        # self.optimizer,self.scheduler=self.getOptimizerAndSheduler()

    '''
    def get_train_test_split(self, test_split, samples):
        split = len(samples) * test_split
        train = samples[0:len(samples) - int(split)]
        test = samples[len(samples) - int(split):-1]
        return train, test

    '''
    '''
    def getDataLoaders(self, data):
        random.shuffle(data)
        train, test = self.get_train_test_split(0.05, data)
        test_dataloader = DataLoader(test, shuffle=True, batch_size=self.params["eval_batch_size"],
                                        collate_fn=self.collator.collate)
        train_dataloader = DataLoader(train, shuffle=True, batch_size=self.params["train_batch_size"],
                                        collate_fn=self.collator.collate)
        return train_dataloader, test_dataloader
    '''
    def getOptimizerAndSheduler(self, len_train_Data):
        optimizer = standard_optimizer.get_bert_optimizer([self.model], self.params["type_optimization"],
            self.params["learning_rate"],
                fp16=self.params.get("fp16"))
        scheduler = standard_optimizer.get_scheduler(self.params, optimizer, len_train_Data)
        return optimizer, scheduler

    def make_forward_pass(self, batch, step):
        context_input=self.collator.collate_context(batch[0])
        candidate_input =self.collator.collate_entities(batch[1])
        labels=torch.tensor(batch[2],device=self.device)
        # label_input = batch[0]["label_idx"].to(device)
        # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
        # context_input, candidate_input, label_input = batch

        loss, logits = self.model(context_input,candidate_input,labels)

        '''
        input=self.collator.collate_batch_train(batch)
        candidate_input = input["candidate_input"]
        context_input = input["context_input"]
        # label_input = batch[0]["label_idx"].to(device)
        # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
        # context_input, candidate_input, label_input = batch

        loss, logits = self.model(context_input,candidate_input)
        '''
        return logits, loss

class TrainerRankerHuggingface:
    def __init__(self, params, evaluate_after_batch,  device):
        self.grad_acc_steps = params["gradient_accumulation_steps"]
        self.params = params
        self.evaluate_after = evaluate_after_batch
        #self.candidate_size = candidate_size
        self.device = device
        self.model = hf_Ranker(params,device=device)
        self.collator = Biencoder_Collator_Huggingface(tokenizer=self.model.tokenizer, args=params,
                                 device=device)

        self.model.to(device)
        # self.optimizer,self.scheduler=self.getOptimizerAndSheduler()

    '''
    def get_train_test_split(self, test_split, samples):
        split = len(samples) * test_split
        train = samples[0:len(samples) - int(split)]
        test = samples[len(samples) - int(split):-1]
        return train, test

    '''
    '''
    def getDataLoaders(self, data):
        random.shuffle(data)
        train, test = self.get_train_test_split(0.05, data)
        test_dataloader = DataLoader(test, shuffle=True, batch_size=self.params["eval_batch_size"],
                                        collate_fn=self.collator.collate)
        train_dataloader = DataLoader(train, shuffle=True, batch_size=self.params["train_batch_size"],
                                        collate_fn=self.collator.collate)
        return train_dataloader, test_dataloader
    '''
    def getOptimizerAndSheduler(self, len_train_Data):
        optimizer = standard_optimizer.get_bert_optimizer([self.model], self.params["type_optimization"],
            self.params["learning_rate"],
                fp16=self.params.get("fp16"))
        scheduler = standard_optimizer.get_scheduler(self.params, optimizer, len_train_Data)
        return optimizer, scheduler

    def make_forward_pass(self, batch, step):
        input=self.collator.collate_batch_train(batch)
        candidate_input = input["candidate_input"]
        context_input = input["context_input"]
        # label_input = batch[0]["label_idx"].to(device)
        # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
        # context_input, candidate_input, label_input = batch

        loss, logits = self.model(context_input,candidate_input)
        return logits, loss
class TrainerE5:
    def __init__(self, params, evaluate_after_batch,  device):
        self.grad_acc_steps = params["gradient_accumulation_steps"]
        self.params = params
        self.evaluate_after = evaluate_after_batch
        #self.candidate_size = candidate_size
        self.device = device
        self.model = E5Ranker(device, params)
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        self.collator = E5collator(tokenizer=self.tokenizer,device=self.device)

        self.model.to(device)
        # self.optimizer,self.scheduler=self.getOptimizerAndSheduler()

    '''
    def get_train_test_split(self, test_split, samples):
        split = len(samples) * test_split
        train = samples[0:len(samples) - int(split)]
        test = samples[len(samples) - int(split):-1]
        return train, test

    '''
    '''
    def getDataLoaders(self, data):
        random.shuffle(data)
        train, test = self.get_train_test_split(0.05, data)
        test_dataloader = DataLoader(test, shuffle=True, batch_size=self.params["eval_batch_size"],
                                        collate_fn=self.collator.collate)
        train_dataloader = DataLoader(train, shuffle=True, batch_size=self.params["train_batch_size"],
                                        collate_fn=self.collator.collate)
        return train_dataloader, test_dataloader
    '''
    def getOptimizerAndSheduler(self, len_train_Data):
        optimizer = standard_optimizer.get_bert_optimizer([self.model], self.params["type_optimization"],
            self.params["learning_rate"],
                fp16=self.params.get("fp16"))
        scheduler = standard_optimizer.get_scheduler(self.params, optimizer, len_train_Data)
        return optimizer, scheduler

    def make_forward_pass(self, batch, step):
        queries = batch[0]
        queries = self.collator.collate(queries, is_passage=False)
        documents = batch[1]

        documents = self.collator.collate(documents, is_passage=True)
        queries.extend(documents)
        token_input = self.tokenizer(queries, max_length=512, padding=True, truncation=True, return_tensors='pt')
        token_input.to(self.device)
        labels = torch.tensor(batch[2], device=self.device)
        # label_input = batch[0]["label_idx"].to(device)
        # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
        # context_input, candidate_input, label_input = batch

        loss, logits = self.model(token_input, len(batch[0]), labels)
        '''
        queries=[sample[1]for sample in batch]
        queries=self.collator.collate(queries,is_passage=False)
        documents = [sample[0] for sample in batch]
        documents=self.collator.collate(documents,is_passage=True)
        queries.extend(documents)

        documents = self.tokenizer(queries, max_length=512, padding=True, truncation=True, return_tensors='pt')
        documents.to(self.device)
        # label_input = batch[0]["label_idx"].to(device)
        # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
        # context_input, candidate_input, label_input = batch

        loss, logits = self.model(documents)
        '''
        return logits, loss