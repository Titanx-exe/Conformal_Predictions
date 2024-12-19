import json
import pickle
from torch.utils.data import DataLoader
#from transformers import AutoTokenizer
import torch
from tqdm import tqdm, trange
from parameters import RankingParser
from models.E5 import E5Ranker
import random
import os
from pytorch_transformers.tokenization_bert import BertTokenizer
from optimizers import noise
import Trainer
#from blinkRanker.parameters import BlinkParser
#from blinkRanker import trainer as blinkTrainer ,evaluator as blinkEvaluator
import data_processing
import Evaluator
import logging
from data_processing import Aida_joint_el
logging.disable(logging.WARNING)
device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")



max_candsize=5
add_gold_mention=True


def load_train_blink_Ranking_Model(device):
    parser = RankingParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)
    params = args.__dict__
    #for lc-quad
    if params["dataset"]=="lcquad":
        entities,documents,doc_to_ent=data_processing.process_lcquad_file("data/train/lcquad.json")
    #for mintaka
    elif params["dataset"]=="mintaka":
        entities, documents, doc_to_ent = data_processing.process_minitaka_file("data/mintaka/mintaka_train.json")

    #for E5
    if params["found_model"]=="e5":
        train_inst = Trainer.TrainerE5(params=params, evaluate_after_batch=params["eval_interval"], device=device)
    # for BiEncoder
    if params["found_model"] == "biencoder":
        train_inst = Trainer.TrainerRanker(params=params, evaluate_after_batch=params["eval_interval"], device=device)

    #for aida
    if params["dataset"] == "aida":
        dp=Aida_joint_el()
        tk = BertTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
        entities, documents, doc_to_ent=dp.read_ds_to_list("data/aida/wikidata/aida_train",tk)
    #entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/aida_train",tk)

    '''
    queries= {}
    indexToNode={}
    for node in handbook:
        if "questions"in node["handbookData"]:
            indexToNode.update({str(node["nodeId"]["clusterId"])+"-"+str(node["nodeId"]["entityId"]):node})
            for question in node["handbookData"]["questions"]:
                if not question in queries:
                    queries.update({question:[]})
                queries[question].append(str(node["nodeId"]["clusterId"])+"-"+str(node["nodeId"]["entityId"]))
    '''
    '''
    train_dataloader = DataLoader(list(entities.keys()), shuffle=True, batch_size=1,
                                  )
    '''
    train_dataloader = DataLoader(list(entities.keys()), shuffle=True, batch_size=1,
                                  )
    #train_inst=Trainer.TrainerRanker(params=params, evaluate_after_batch=params["eval_interval"], device=device)

    optimizer, scheduler = train_inst.getOptimizerAndSheduler(len(entities))
    #for lcquad
    #evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator)
    #for minaka
    #evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,use_lcquad=False)

    #for aida
    #eval_entities, eval_documents, eval_doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/aida_testa",tk )
    '''
    evaluator_inst = Evaluator.IndexEvaluator(params=params,collator=train_inst.collator,entities=eval_entities
                                              ,doc_to_ent=eval_doc_to_ent,documents=eval_documents)
                                              
    '''
    #filehandler=data_processing.process_lcquad_file
    if params["dataset"] == "aida":
        tk = BertTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,filehandler=dp.read_ds_to_list,file="data/aida/wikidata/aida_testa",tokenizer=tk)
    if params["dataset"] == "mintaka":
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,
                                              filehandler=data_processing.process_minitaka_file, file="data/mintaka/mintaka_test.json")
    if params["dataset"] == "lcquad":
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,
                                                  filehandler=data_processing.process_lcquad_file, file="data/test/lcquad.json")
    evaluator_inst.entities.extend(entities.keys())
    return train_inst,evaluator_inst, train_dataloader, optimizer,scheduler,entities,documents,doc_to_ent


def save_model(model, tokenizer, output_dir):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(output_dir, "model_config")
    torch.save(model_to_save.state_dict(), output_model_file)
    #model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def encode_documents(documents,model,collator):
    documents=list(documents)
    data_loader = DataLoader(documents, shuffle=True, batch_size=100,
                             collate_fn=collator.collate_context)
    #iter_ = tqdm(data_loader, desc="Encode Train Documents")
    doc_encodings = []
    for step, batch in enumerate(data_loader):
        if not isinstance(model,E5Ranker):
            context_input = batch["context_input"]
            # candidate_input = batch["candidate_input"]
            # labels=[0 for i in range(batch["candidate_input"].size(0))]
            # label_input = batch[0]["label_idx"].to(device)
            # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
            # context_input, candidate_input, label_input = batch
            encodings = model.encode_context(context_input).tolist()
        else:
            encodings=model.encode_context(batch).tolist()
        doc_encodings.extend(encodings)
    encoding_map={}
    for i in range(len(doc_encodings)):
        encoding_map[documents[i]]=doc_encodings[i]
    return encoding_map

def train(epochs):
    #trainer,evaluator, train_dataloader, optimizer, scheduler = load_train_only_Graph_Model(device)
    trainer, evaluator, train_dataloader, optimizer, scheduler,entities,documents,doc_to_ent = load_train_blink_Ranking_Model(device)
    trainer.model.train()
    #print(evaluator.evaluate(trainer.model))
    index,results=evaluator.evaluate(trainer.model)
    print(results)
    encoding_map=encode_documents(documents,trainer.model,trainer.collator)
    for e in range(epochs):
        num_batch = 0

        # step=0
        iter_ = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(iter_):
            #batch=data_processing.create_batch_ent(batch[0],list(entities[batch[0]]),random.sample(list(documents),1000),doc_to_ent)
            batch=data_processing.create_batch_index(batch[0],entities,list(entities[batch[0]]),encoding_map,index,doc_to_ent)
            #batch = data_processing.create_batch_index_document(batch[0], entities,  encoding_map,
            #                                           index, doc_to_ent)
            logits, loss = trainer.make_forward_pass(batch,step)

            if trainer.grad_acc_steps > 1:
                loss = loss / trainer.grad_acc_steps
            loss.backward()
            if (step + 1) % trainer.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), trainer.params["max_grad_norm"]
                )
                #noise.add_anticorrelated_noise_prev_term(optimizer, device)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            num_batch += 1
            # print("Train Epoch " + str(e), "batch " + str(num_batch) + " Loss: " + str(loss.item()))
            if num_batch % trainer.evaluate_after == 0:
                print("Start evaluation in epoch:" + str(e) + " batch: " + str(num_batch))
                trainer.model.eval()
                print(evaluator.evaluate(trainer.model))
                index,results = evaluator.evaluate(trainer.model)
                print(results)
                encoding_map = encode_documents(documents, trainer.model, trainer.collator)
                #epoch_output_folder_path = os.path.join(
                #    "ranker_gr", "epoch_{}_{}".format(e, num_batch))
                #save_model(model, model.tokenizer, epoch_output_folder_path)
                trainer.model.train()

        print("Start evaluation after epoch: " + str(e))
        trainer.model.eval()
        index,results = evaluator.evaluate(trainer.model)
        print("---------------------------Results in Epoch------------------------:" + str(e))
        print(results)
        #Recall writing in a file
        f = open('Results_Recall.txt', 'a+')
        f.write("Results in Epoch: " + str(e)+ str(results)+ '\n')
        f.close()
        encoding_map = encode_documents(documents, trainer.model, trainer.collator)
        epoch_output_folder_path = os.path.join(
            "ranker_lcquad_test_noise", "epoch_{}".format(e)
        )
        save_model(trainer.model,trainer.tokenizer,  epoch_output_folder_path)
        trainer.model.train()
train(10)