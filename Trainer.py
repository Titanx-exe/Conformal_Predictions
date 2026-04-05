import pickle
from optimizers import standard_optimizer
import torch
#from blinkRanker import collator
from models.BiEncoder import BiEncoderRanker
from models.BiEncoderHuggingface import BiEncoderRanker as hf_Ranker
from models.E5 import E5Ranker
from models.qwen3 import Qwen3Ranker
from models.llama3 import Llama3Ranker
import random
from collator import Biencoder_Collator,Biencoder_Collator_Huggingface,E5collator, Qwen3Collator, Llama3Collator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from models.Llama3_LBW import Llama3LBWRanker
from collator import Llama3LBWCollator
from models.llama_decoder import LlamaDecoderRanker
from collator import LlamaDecoderCollator
from models.qwen3_decoder import Qwen3DecoderRanker
from collator import Qwen3DecoderCollator

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
        self.tokenizer = self.model.tokenizer
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
        #self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
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



class TrainerQwen3:
    def __init__(self, params, evaluate_after_batch, device):
        self.grad_acc_steps = params["gradient_accumulation_steps"]
        self.params = params
        self.evaluate_after = evaluate_after_batch
        #self.candidate_size = candidate_size
        self.device = device
        
        self.model = Qwen3Ranker(device, params)
        self.tokenizer =  self.model.tokenizer
        self.collator = Qwen3Collator(tokenizer=self.tokenizer, device=self.device)
        self.model.to(device)
        
#        if self.params['found_model'] == 'qwen3':
 #           print("LOADING QWEN3 TOKENIZER")
  #          self.tokenizer = AutoTokenizer.from_pretrained(
   #             'Qwen/Qwen3-Embedding-0.6B',
    #            padding_side='left'
     #       )
      #  elif self.params['found_model'] == 'dual_encoder':
       #     print("LOADING dual-encoder TOKENIZER")
        #    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        ##########################################################
        
       # self.collator = E5collator(tokenizer=self.tokenizer, device=self.device)
        #self.model.to(device)

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
        
        # Forward pass - identical interface to E5Ranker
        loss, logits = self.model(token_input, len(batch[0]), labels)
        ##########################################################
        
        '''
        # Alternative implementation (commented out, same as TrainerE5)
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

class TrainerLlama3:
    def __init__(self, params, evaluate_after_batch, device):
        self.grad_acc_steps = params["gradient_accumulation_steps"]
        self.params = params
        self.evaluate_after = evaluate_after_batch
        self.device = device
        
        # Initialize model (tokenizer now created inside Llama3Ranker)
        self.model = Llama3Ranker(device, params)
        
        # CRITICAL FIX: Get tokenizer from model
        self.tokenizer = self.model.tokenizer
        
        # CRITICAL FIX: Use Llama3Collator instead of E5collator
        self.collator = Llama3Collator(tokenizer=self.tokenizer, device=self.device)
        
        # Move model to device
        self.model.to(device)
        
        print(f"✓ Llama3Ranker initialized on {device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e9:.2f}B")
    
    def getOptimizerAndSheduler(self, len_train_Data):
        """Initialize optimizer and scheduler"""
        optimizer = standard_optimizer.get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16")
        )
        scheduler = standard_optimizer.get_scheduler(self.params, optimizer, len_train_Data)
        return optimizer, scheduler
    
    def make_forward_pass(self, batch, step):
        """
        Make forward pass through Llama3 model.

        Args:
            batch: Tuple of (queries, documents, labels)
            step: Current training step

        Returns:
            logits: Similarity scores
            loss: Computed loss
        """
        # CRITICAL FIX: Store original counts BEFORE any processing
        original_query_count = len(batch[0])
        original_doc_count = len(batch[1])
        
        # Extract batch components
        queries = batch[0]
        queries = self.collator.collate(queries, is_passage=False)

        documents = batch[1]
        documents = self.collator.collate(documents, is_passage=True)

        # Concatenate queries and documents
        queries.extend(documents)

        # CRITICAL FIX: Use reduced max_length for decoder models (memory efficiency)
        # Llama3-8B with 512 tokens can cause OOM; use 128 or 256
        max_length = self.params.get('max_seq_length', 128)

        # Tokenize
        token_input = self.tokenizer(
            queries,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        token_input = {k: v.to(self.device) for k, v in token_input.items()}

        # Extract labels
        labels = torch.tensor(batch[2], device=self.device)

        # Forward pass - use ORIGINAL query count, not current length
        loss, logits = self.model(token_input, original_query_count, labels)

        return logits, loss

class TrainerLlama3LBW:
    def __init__(self, params, evaluate_after_batch, device):
        self.grad_acc_steps = params["gradient_accumulation_steps"]
        self.params = params
        self.evaluate_after = evaluate_after_batch
        self.device = device

        self.model = Llama3LBWRanker(device, params)
        self.tokenizer = self.model.tokenizer
        self.collator = Llama3LBWCollator(tokenizer=self.tokenizer, device=self.device)
        self.model.to(device)

    def getOptimizerAndSheduler(self, len_train_Data):
        # NOTE: Using the new optimizer key 'all_encoder_layers_llama3_lbw'
        optimizer = standard_optimizer.get_bert_optimizer(
            [self.model], 
            'all_encoder_layers_llama3_lbw', # Hardcoded or passed via params["type_optimization"]
            self.params["learning_rate"],
            fp16=self.params.get("fp16")
        )
        scheduler = standard_optimizer.get_scheduler(self.params, optimizer, len_train_Data)
        return optimizer, scheduler

    def make_forward_pass(self, batch, step):
        queries = batch[0]
        queries = self.collator.collate(queries, is_passage=False)
        documents = batch[1]
        documents = self.collator.collate(documents, is_passage=True)

        original_query_count = len(queries)
        combined_text = queries + documents
        
        token_input = self.tokenizer(
            combined_text, max_length=128, padding=True, truncation=True, return_tensors='pt'
        )
        token_input = {k: v.to(self.device) for k, v in token_input.items()}
        labels = torch.tensor(batch[2], device=self.device)

        loss, logits = self.model(token_input, original_query_count, labels)
        return logits, loss

class TrainerLlamaDecoder:
    def __init__(self, params, evaluate_after_batch, device):
        self.grad_acc_steps = params["gradient_accumulation_steps"]
        self.params = params
        self.evaluate_after = evaluate_after_batch
        self.device = device
        
        self.model = LlamaDecoderRanker(device, params)
        self.tokenizer = self.model.tokenizer
        self.collator = LlamaDecoderCollator(tokenizer=self.tokenizer, device=self.device)
        self.model.to(device)
    
    def getOptimizerAndSheduler(self, len_train_Data):
        optimizer = standard_optimizer.get_bert_optimizer(
            [self.model],
            'all_encoder_layers_llama_decoder',  # Need to add this pattern
            self.params["learning_rate"],
            fp16=self.params.get("fp16")
        )
        scheduler = standard_optimizer.get_scheduler(self.params, optimizer, len_train_Data)
        return optimizer, scheduler
    
    def make_forward_pass(self, batch, step):
        queries = batch[0]
        queries = self.collator.collate(queries, is_passage=False)
        
        documents = batch[1]
        documents = self.collator.collate(documents, is_passage=True)
        
        original_query_count = len(queries)
        combined_text = queries + documents
        
        token_input = self.tokenizer(
            combined_text, max_length=128, padding=True, truncation=True, return_tensors='pt'
        )
        token_input = {k: v.to(self.device) for k, v in token_input.items()}
        labels = torch.tensor(batch[2], device=self.device)
        
        loss, logits = self.model(token_input, original_query_count, labels)
        return logits, loss


class TrainerQwen3Decoder:
    """Trainer for Qwen3 Decoder model with Look-Both-Ways support."""
    def __init__(self, params, evaluate_after_batch, device):
        self.grad_acc_steps = params["gradient_accumulation_steps"]
        self.params = params
        self.evaluate_after = evaluate_after_batch
        self.device = device
        
        self.model = Qwen3DecoderRanker(device, params)
        self.tokenizer = self.model.tokenizer
        self.collator = Qwen3DecoderCollator(tokenizer=self.tokenizer, device=self.device)
        self.model.to(device)
    
    def getOptimizerAndSheduler(self, len_train_Data):
        optimizer = standard_optimizer.get_bert_optimizer(
            [self.model],
            'all_encoder_layers_qwen3_decoder',
            self.params["learning_rate"],
            fp16=self.params.get("fp16")
        )
        scheduler = standard_optimizer.get_scheduler(self.params, optimizer, len_train_Data)
        return optimizer, scheduler
    
    def make_forward_pass(self, batch, step):
        queries = batch[0]
        queries = self.collator.collate(queries, is_passage=False)
        
        documents = batch[1]
        documents = self.collator.collate(documents, is_passage=True)
        
        original_query_count = len(queries)
        combined_text = queries + documents
        
        token_input = self.tokenizer(
            combined_text, max_length=128, padding=True, truncation=True, return_tensors='pt'
        )
        token_input = {k: v.to(self.device) for k, v in token_input.items()}
        labels = torch.tensor(batch[2], device=self.device)
        
        loss, logits = self.model(token_input, original_query_count, labels)
        return logits, loss
