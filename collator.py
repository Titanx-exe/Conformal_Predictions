import pickle

import torch
from tqdm import tqdm, trange

#from generic_training import save_model


class RankingCollator:
    def __init__(self, tokenizer, max_context_length,max_cand_length, device):
        self.device=device
        self.tokenizer = tokenizer
        self.candidateMap={}
        self.candidateEncodings={}
        self.max_context_length=max_context_length
        self.max_cand_length = max_cand_length


    def select_field(self, data, key1, key2=None):
        if key2 is None:
            return [example[key1] for example in data]
        else:
            if key1 == "candidates":
                label_ids = []
                for example in data:
                    candidates = []
                    candidates.append([cand["ids"] for cand in example["candidates"]])
                    label_ids.extend(candidates)
                return label_ids
            return [example[key1][key2] for example in data]
    def get_candidate_representation(
            self,
            candidate_text,
            context_tokens

    ):
        sep_token = self.tokenizer.sep_token
        cand_tokens = self.tokenizer.tokenize(candidate_text)

        cand_tokens = cand_tokens[: self.max_cand_length - 2]
        cand_tokens = context_tokens + cand_tokens
        if len(cand_tokens)>self.max_cand_length-1:
            cand_tokens=cand_tokens[0:self.max_cand_length-1]
        cand_tokens = cand_tokens + [sep_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(cand_tokens)
        padding = [0] * (self.max_cand_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == self.max_cand_length

        return {
            "tokens": cand_tokens,
            "ids": input_ids,
        }

    def get_context_tokens(
            self,
            query

    ):

        tokens = self.tokenizer.tokenize(query)

        context_tokens = ["[CLS]"] + tokens + ["[SEP]"]

        return context_tokens

    def process_mention_data(
            self,
            samples,

    ):
        processed_samples = []


        iter_ = tqdm(samples)

        # use_world = True

        for idx, sample in enumerate(iter_):
            context_tokens = self.get_context_tokens(
                sample["query"]
            )
            candidate_representaions = []
            candidates = []
            #candidates.append(sample["entity"])
            candidates.extend(sample["candidates"])
            for cand in candidates:
                label_tokens = self.get_candidate_representation(
                    cand,context_tokens)
                candidate_representaions.append(label_tokens)
            # label_idx = int(self.label[sample["label"]])

            record = {
                "candidates": candidate_representaions,
            }
            # use_world=False
            processed_samples.append(record)


        cand_vecs = torch.tensor(
            self.select_field(processed_samples, "candidates", "ids"), dtype=torch.long,
        )

        return cand_vecs
    def collate(self,batch):

        candidates=self.process_mention_data(batch)
        labels=[]
        for el in batch: labels.append(el["label_id"])
        label_tensor=torch.LongTensor(labels)
        return {"candidate_encodings":candidates.to(self.device), "labels":label_tensor.to(self.device)}


    def process_mention_data_prod(
            self,
            query,
            candidates,

    ):

        context_tokens = self.get_context_tokens(
                query
        )
        candidate_representaions = []
        # candidates.append(sample["entity"])
        for cand in candidates:
            sep_token = self.tokenizer.sep_token
            cand_tokens = self.tokenizer.tokenize(cand)

            cand_tokens = cand_tokens[: self.max_cand_length - 2]
            cand_tokens = context_tokens + cand_tokens
            if len(cand_tokens) > self.max_cand_length - 1:
                cand_tokens = cand_tokens[0:self.max_cand_length - 1]
            cand_tokens = cand_tokens + [sep_token]

            input_ids = self.tokenizer.convert_tokens_to_ids(cand_tokens)
            padding = [0] * (self.max_cand_length - len(input_ids))
            input_ids += padding
            assert len(input_ids) == self.max_cand_length
            candidate_representaions.append(torch.tensor(input_ids))

        return candidate_representaions



class Biencoder_Collator_Huggingface():
    def __init__(self,tokenizer,args,device="cpu"):
        self.tokenizer = tokenizer
        self.args = args
        self.device=device
        self.entity_text_dict=pickle.load(open("data/entity_descriptions.pkl","rb"))


    def collate_entities(self,batch):
        candidates = ["[CLS]" + cand if not cand in self.entity_text_dict
                            else "[CLS]" + self.entity_text_dict[cand] for cand in batch]
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
        questions=["[CLS]" + question for question in batch]
        batch = self.tokenizer(questions, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return batch.to(self.device)


class Biencoder_Collator():
    def __init__(self,tokenizer,args,device="cpu"):
        self.tokenizer = tokenizer
        self.args = args
        self.device=device
        self.entity_text_dict=pickle.load(open("data/entity_descriptions.pkl","rb"))

    def process_sample(self, question, candidates=None, label=None):
        question_tokens = self.tokenizer.tokenize("[CLS]"+question)
        question_ids = self.tokenizer.convert_tokens_to_ids(question_tokens)
        if len(question_ids) > self.args["max_context_length"]:
            cans = question_ids[0:self.args["max_context_length"] - 1]
            cans.append(question_ids[len(question_ids) - 1])
            question_ids = cans
        padding = [0] * (self.args["max_context_length"] - len(question_ids))
        question_ids += padding

        if candidates is not None:
            candidate_tokens = []
            for cand in candidates:
                if not cand in self.entity_text_dict:
                    cand_tokens = self.tokenizer.tokenize("[CLS]"+cand)
                else:
                    cand_tokens = self.tokenizer.tokenize("[CLS]" + self.entity_text_dict[cand])
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
    def collate_entities(self,batch):
        candidate_tokens = []
        for cand in batch:
            if not cand in self.entity_text_dict:
                cand_tokens = self.tokenizer.tokenize("[CLS]" + cand)
            else:
                cand_tokens = self.tokenizer.tokenize("[CLS]" + self.entity_text_dict[cand])
            cand_ids = self.tokenizer.convert_tokens_to_ids(cand_tokens)
            if len(cand_ids) > self.args["max_cand_length"]:
                cans = cand_ids[0:self.args["max_cand_length"] - 1]
                cans.append(cand_ids[len(cand_ids) - 1])
                cand_ids = cans
            padding = [0] * (self.args["max_cand_length"] - len(cand_ids))
            cand_ids += padding
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
        return torch.tensor(question_input, device=self.device)
class E5collator:
    def __init__(self,tokenizer,device):
        self.tokenizer=tokenizer
        self.entity_text_dict = pickle.load(open("data/ent_descriptions_update.pkl", "rb"))
        self.device=device
    def collate(self,batch, is_passage):
        if is_passage:
            repr=["passage: "+ self.entity_text_dict[text]if text in self.entity_text_dict else "passage: "+ text for text in batch]
        else:
            repr=["query: "+text for text in batch]
        return repr
    def collate_entities(self,batch):
        repr=["passage: "+ self.entity_text_dict[text]if text in self.entity_text_dict else "passage: "+ text for text in batch]
        batch=self.tokenizer(repr, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return batch.to(self.device)

    def collate_context(self,batch):
        repr = ["query: " + text for text in batch]
        batch = self.tokenizer(repr, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return batch.to(self.device)
