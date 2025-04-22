import json
import pickle
from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
import torch
from optimizers import standard_optimizer
from tqdm import tqdm, trange
from parameters import RankingParser
from models.E5 import E5Ranker
from models.BiEncoderHuggingface import BiEncoderRanker
import random
import os
from pytorch_transformers.tokenization_bert import BertTokenizer
from optimizers import noise
# from blinkRanker.parameters import BlinkParser
# from blinkRanker import trainer as blinkTrainer ,evaluator as blinkEvaluator
import data_processing
import Evaluator
import logging
from ms_marco_data_handler import Ms_marco_data_handler
from MS_marco_collator import Biencoder_Collator,E5collator
from data_processing import Aida_joint_el
from transformers import AutoTokenizer

logging.disable(logging.WARNING)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

max_candsize = 5
add_gold_mention = True

class TrainerRankerHuggingface:
    def __init__(self, params, evaluate_after_batch,handler, device):
        self.grad_acc_steps = params["gradient_accumulation_steps"]
        self.params = params
        self.evaluate_after = evaluate_after_batch
        #self.candidate_size = candidate_size
        self.device = device
        self.model = BiEncoderRanker(params,device=device)
        self.tokenizer = self.model.tokenizer
        self.collator = Biencoder_Collator(tokenizer=self.model.tokenizer,args=params
                                           ,queries=handler.queries, device=device)

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

    def make_forward_pass(self, batch):
        context_input=self.collator.collate_context(batch[0])
        candidate_input =self.collator.collate_entities(batch[1])
        labels=torch.tensor(batch[2],device=self.device)
        # label_input = batch[0]["label_idx"].to(device)
        # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
        # context_input, candidate_input, label_input = batch

        loss, logits = self.model(context_input,candidate_input,labels)
        return logits, loss
class TrainerE5:
    def __init__(self, params, evaluate_after_batch,handler,  device):
        self.grad_acc_steps = params["gradient_accumulation_steps"]
        self.params = params
        self.evaluate_after = evaluate_after_batch
        #self.candidate_size = candidate_size
        self.device = device
        self.model = E5Ranker(device, params)
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        self.collator = E5collator(tokenizer=self.tokenizer,queries=handler.queries,device=self.device)

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

    def make_forward_pass(self, batch):
        queries= batch[0]
        queries=self.collator.collate(queries,is_passage=False)
        documents = batch[1]

        documents=self.collator.collate(documents,is_passage=True)
        queries.extend(documents)
        token_input=self.tokenizer(queries, max_length=512, padding=True, truncation=True, return_tensors='pt')
        token_input.to(self.device)
        labels = torch.tensor(batch[2], device=self.device)
        # label_input = batch[0]["label_idx"].to(device)
        # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
        # context_input, candidate_input, label_input = batch

        loss, logits = self.model(token_input,len(batch[0]),labels)
        return logits, loss

def handle_eval_file(queries,eval_documents="data/msmarco/eval_documents"):
    documents=set()
    queries=pickle.load(open(queries,"rb"))
    doc_to_ent={}
    for query in queries:
        documents.add(query)
        doc_to_ent[query]=queries[query]["relevant"]
    entities=pickle.load(open(eval_documents,"rb"))
    return entities,documents,doc_to_ent

def load_train_blink_Ranking_Model():
    parser = RankingParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)
    params = args.__dict__

    global device
    device = torch.device(
        "cuda:" + str(params["gpu_id"]) if torch.cuda.is_available() else "cpu")
    # for lc-quad
    handler = Ms_marco_data_handler(params)
    # for E5
    if params["found_model"] == "e5":
        train_inst = TrainerE5(params=params, evaluate_after_batch=params["eval_interval"],handler=handler, device=device)
    # for BiEncoder
    if params["found_model"] == "biencoder":
        train_inst = TrainerRankerHuggingface(params=params, evaluate_after_batch=params["eval_interval"],handler=handler,
                                                      device=device)
    eval_queries=pickle.load(open("data/msmarco/eval_queries","rb"))
    eval_documents=pickle.load(open("data/msmarco/eval_documents","rb"))
    eval_documents={el: "title: " + eval_documents[el][1] + "[SEP] content: " + eval_documents[el][2] for el in eval_documents}
    eval_collator=Biencoder_Collator(train_inst.collator.tokenizer,params,eval_queries,device=device)
    eval_collator.documents=eval_documents
    evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=eval_collator,
                                              filehandler=handle_eval_file,
                                              file="data/msmarco/eval_queries")

    #train_dataloader = DataLoader(list(entities.keys()), shuffle=True, batch_size=1,
    #                              )
    # train_inst=Trainer.TrainerRanker(params=params, evaluate_after_batch=params["eval_interval"], device=device)

    optimizer, scheduler = train_inst.getOptimizerAndSheduler(10000)


    return train_inst, evaluator_inst, optimizer, scheduler, handler


def save_model(model, tokenizer, output_dir):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(output_dir, "model_config")
    torch.save(model_to_save.state_dict(), output_model_file)
    # model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)





def train():
    # trainer,evaluator, train_dataloader, optimizer, scheduler = load_train_only_Graph_Model(device)
    trainer, evaluator, optimizer, scheduler,handler = load_train_blink_Ranking_Model()
    trainer.model.train()
    epochs = trainer.params['num_epochs']
    # print(evaluator.evaluate(trainer.model))
    _, results, mrr = evaluator.evaluate(trainer.model)
    # results, mrr = evaluator.evaluate_mrr(trainer.model)
    print(results)
    print(mrr)
    f = open(trainer.params["training_result_update_file"], 'a+')
    f.write("####Noise ratio taken as: " + ' ' + str(trainer.params['noise_ratio']) + '\n'
            + "##Smoothing factor taken as: " + ' ' + str(trainer.params['label_smoothness']) + '\n')
    f.close()
    # writing mrrs
    f1 = open('Results_Mrr.txt', 'a+')
    f1.write("####Noise ratio taken as: " + ' ' + str(trainer.params['noise_ratio']) + '\n'
             + "##Smoothing factor taken as: " + ' ' + str(trainer.params['label_smoothness']) + '\n')
    f1.close()
    #encoding_map = encode_documents(documents, trainer.model, trainer.collator)
    base_smoothing = float(trainer.params['base_smoothing_rate'])
    avg_loss = []
    handler.reload_current_data(trainer.model,trainer.collator,reload_full=True)
    for e in range(epochs):
        total_loss = 0.0
        final_output = 0
        avg_epoch_loss = 0.0
        EPOCH_FILE = "epoch.txt"
        with open(EPOCH_FILE, "w") as f:
            f.write(str(e))
        num_batch = 0
        # step=0
        iter_ = tqdm(range(trainer.params["training_steps_per_split"]), desc="Training")
        for step, batch in enumerate(iter_):
            # batch=data_processing.create_batch_ent(batch[0],list(entities[batch[0]]),random.sample(list(documents),1000),doc_to_ent)
            batch = handler.create_batch_index(num_noise_labels=trainer.params['noise_ratio'])
            # batch = data_processing.create_batch_index_document(batch[0], entities,  encoding_map,
            #                                           index, doc_to_ent)
            logits, loss = trainer.make_forward_pass(batch)
            total_loss = total_loss + loss.item()  # Adding up the loss
            if trainer.grad_acc_steps > 1:
                loss = loss / trainer.grad_acc_steps
            loss.backward()
            if (step + 1) % trainer.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), trainer.params["max_grad_norm"]
                )
                noise_function = trainer.params["noise_approach"]
                if noise_function == "anticorrelated_noise_prev_term":
                    noise.add_anticorrelated_noise_prev_term(optimizer, device)
                if noise_function == "gausian_noise":
                    noise.add_gausian_noise(optimizer, device)
                if noise_function == "anticorrelated_noise_gradient":
                    noise.add_anticorrelated_noise_gradient(optimizer, device)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            num_batch += 1
            # print("Train Epoch " + str(e), "batch " + str(num_batch) + " Loss: " + str(loss.item()))
            #if num_batch % trainer.evaluate_after == 0:
            if float(num_batch) == trainer.params["training_steps_per_split"]/2:
                print("Start evaluation in epoch:" + str(e) + " batch: " + str(num_batch))
                trainer.model.eval()
                #print(evaluator.evaluate(trainer.model))
                # print(evaluator.evaluate_mrr(trainer.model))
                _, results, mrr = evaluator.evaluate(trainer.model)
                handler.reload_current_data(trainer.model, trainer.collator)
                # index, mrr = evaluator.evaluate_mrr(trainer.model)
                print(results)
                print(mrr)
                #encoding_map = encode_documents(documents, trainer.model, trainer.collator)
                # epoch_output_folder_path = os.path.join(
                #    "ranker_gr", "epoch_{}_{}".format(e, num_batch))
                # save_model(model, model.tokenizer, epoch_output_folder_path)
                trainer.model.train()
        avg_epoch_loss = total_loss / len(range(trainer.params["training_steps_per_split"]))
        print("Start evaluation after epoch: " + str(e))
        trainer.model.eval()
        index, results, mrr = evaluator.evaluate(trainer.model)
        # index, mrr = evaluator.evaluate_mrr(trainer.model)
        print("---------------------------Results in Epoch------------------------:" + str(e))
        print(results)
        avg_loss.append(avg_epoch_loss)  # Store epoch loss for next iteration
        f = open('loss_file.txt', 'a+')
        f.write("Smoothing factor taken as: " + ' ' + str(trainer.params["label_smoothness"]) + '\n'
                + "Average loss in this epoch" + ' ' + str(avg_epoch_loss) + '\n'
                + "Average loss in the previous epoch:" + ' ' + str(avg_loss[e - 1]) + '\n'
                + "Total loss so far is:" + ' ' + str(avg_loss) + '\n')
        f.close()
        # Recall writing in a file
        f = open(trainer.params["training_result_update_file"], 'a+')
        f.write("Results in Epoch " + str(e) + ' : ' + str(results) + '\n')
        f.close()
        # writing mrrs
        f1 = open('Results_Mrr.txt', 'a+')
        f1.write("Results in Epoch " + str(e) + ' : ' + str(mrr) + '\n')
        f1.close()
        #encoding_map = encode_documents(documents, trainer.model, trainer.collator)
        epoch_output_folder_path = os.path.join(
            trainer.params["model_dump_folder"], "epoch_{}".format(e)
        )
        save_model(trainer.model,trainer.tokenizer,  epoch_output_folder_path)
        trainer.model.train()


#train(10)