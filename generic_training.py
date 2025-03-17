import json
import pickle

from sqlalchemy.testing.util import total_size
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
#from data_processing import apply_label_smoothing  # Import the function
logging.disable(logging.WARNING)
device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")



max_candsize=5
add_gold_mention=True


def load_train_blink_Ranking_Model(label_smoothness, epochs):
    parser = RankingParser(add_model_args=True)
    parser.add_training_args()

    parser.add_eval_args()
    parser.add_argument("--label_smoothness", type=float, default=label_smoothness)
    parser.add_argument("--epochs", type=int, default=epochs)

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print("####################################### args:", args)
    params = args.__dict__
    global device
    device=torch.device(
            "cuda:"+str(params["gpu_id"]) if torch.cuda.is_available() else "cpu")
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


def adaptive_smoothing_loss(prev_loss, current_loss, base_smoothing=0.05, min_smoothing=-2.0, max_smoothing=0.3):
    """
    Adjusts label smoothing dynamically based on loss trends.

    :param prev_loss: Loss from the previous epoch
    :param current_loss: Loss from the current epoch
    :param base_smoothing: Default smoothing factor
    :param min_smoothing: Minimum smoothing factor (negative for NLS)
    :param max_smoothing: Maximum smoothing factor (positive for PLS)
    :return: Adjusted smoothing factor
    """
    loss_change = current_loss - prev_loss  # Difference between epochs
    print("***************** previous loss is ***************************", prev_loss)
    print("***************** current loss is ***************************", current_loss)
    if loss_change > 0.1:  # Large loss increase → Increase PLS more aggressively
        smoothing_factor = min(base_smoothing + 0.2, max_smoothing)  # Increase rapidly
    elif loss_change > 0.05:  # Moderate loss increase
        smoothing_factor = min(base_smoothing + 0.1, max_smoothing)
    elif loss_change < -0.1:  # Large loss decrease
        smoothing_factor = max(base_smoothing - 0.2, min_smoothing)
    elif loss_change < -0.05:  # Moderate loss decrease
        smoothing_factor = max(base_smoothing - 0.5, min_smoothing)
    else:  # Loss is stable
        #smoothing_factor = max(base_smoothing + random.uniform(-0.05, 0.05), min_smoothing)
        smoothing_factor = max(base_smoothing - 0.2, min_smoothing)

    return smoothing_factor


def train(epochs, label_smoothness=0.0):
    
    #trainer,evaluator, train_dataloader, optimizer, scheduler = load_train_only_Graph_Model(device)
    trainer, evaluator, train_dataloader, optimizer, scheduler,entities,documents,doc_to_ent = load_train_blink_Ranking_Model(label_smoothness, epochs)

    trainer.model.train()
    prev_loss = float("inf")
    #print(evaluator.evaluate(trainer.model))
    index,results=evaluator.evaluate(trainer.model)
    index, mrr = evaluator.evaluate_mrr(trainer.model)
    print(results)
    print(mrr)
    encoding_map=encode_documents(documents,trainer.model,trainer.collator)


    base_smoothing = float(trainer.params['base_smoothing_rate'])
    smoothing_factor = label_smoothness
    avg_loss = []
    f = open(trainer.params["training_result_update_file"], 'a+')
    f.write("Smoothing factor taken as: " + ' ' + str(smoothing_factor) + '\n')
    f.close()
    f = open("Results_Mrr.txt", 'a+')
    f.write("Smoothing factor taken as: " + ' ' + str(smoothing_factor) + '\n')
    f.close()
    for e in range(epochs):
        num_batch = 0
        total_loss = 0.0
        final_output = 0
        print("Updated learning rate is ---------------::::::::::::::::::::", trainer.params["learning_rate"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = trainer.params["learning_rate"]

        avg_epoch_loss = 0.0

        # step=0
        iter_ = tqdm(train_dataloader, desc="Training")
        EPOCH_FILE = "epoch.txt"
        with open(EPOCH_FILE, "w") as f:
            f.write(str(e))
        for step, batch in enumerate(iter_):
            trainer.params["label_smoothness"] = smoothing_factor  # Use current smoothing factor
            #batch=data_processing.create_batch_ent(batch[0],list(entities[batch[0]]),random.sample(list(documents),1000),doc_to_ent)
            batch=data_processing.create_batch_index(batch[0],entities,list(entities[batch[0]]),encoding_map,index,doc_to_ent)
            #batch = data_processing.create_batch_index_document(batch[0], entities,  encoding_map,
            #                                           index, doc_to_ent)

            if trainer.params['adaptive_label_smoothing'] == 'yes':
                if e == 0:
                    trainer.params["label_smoothness"] = base_smoothing
                else:
                    trainer.params["label_smoothness"] = smoothing_factor  # Use current smoothing factor
            else:
                trainer.params["label_smoothness"] = label_smoothness
                smoothing_factor = label_smoothness

            logits, loss = trainer.make_forward_pass(batch,step)
            #f = open('loss_file.txt', 'a+')
            #f.write("Epoch: " + ' ' + str(e) + '\n')
            #f.write("forward loss is: " + ' ' + str(loss.item()) + '\n')
            #f.close()
            total_loss = total_loss + loss.item() #Adding up the loss

            if torch.isnan(loss):
                print("Warning: total_loss became NaN! Resetting to 0.0")
                total_loss = 0.0
            # Adaptive smoothing based on loss trend
            #current_loss = total_loss / (step + 1)


            if trainer.grad_acc_steps > 1:
                loss = loss / trainer.grad_acc_steps
            loss.backward()
            # Compute total gradient norm
            total_grad_norm = torch.sqrt(
                sum(p.grad.norm() ** 2 for p in trainer.model.parameters() if p.grad is not None))
            #print(f"Total Gradient Norm at Step {step}: {total_grad_norm.item()}")
            #f = open('gradients.txt', 'a+')
            #f.write("Total Gradient Norm at Step " + str(step) + ' ' + str(total_grad_norm.item()) + '\n')
            #f.close()

            if (step + 1) % trainer.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), trainer.params["max_grad_norm"]
                )
                noise_function=trainer.params["noise_approach"]
                if noise_function=="anticorrelated_noise_prev_term":
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
            if num_batch % trainer.evaluate_after == 0:
                print("Start evaluation in epoch:" + str(e) + " batch: " + str(num_batch))
                trainer.model.eval()
                print(evaluator.evaluate(trainer.model))
                print(evaluator.evaluate_mrr(trainer.model))
                index,results = evaluator.evaluate(trainer.model)
                index, mrr = evaluator.evaluate_mrr(trainer.model)
                print(results)
                print(mrr)
                final_output = results
                encoding_map = encode_documents(documents, trainer.model, trainer.collator)
                #epoch_output_folder_path = os.path.join(
                #    "ranker_gr", "epoch_{}_{}".format(e, num_batch))
                #save_model(model, model.tokenizer, epoch_output_folder_path)
                trainer.model.train()

        # **Epoch-Level Loss Computation**
        avg_epoch_loss = total_loss / len(train_dataloader)
        # **Update smoothing factor AFTER the epoch completes**
        if trainer.params['adaptive_label_smoothing'] == 'yes':
            if avg_loss:  # Ensure there's a previous loss recorded
                smoothing_factor = adaptive_smoothing_loss(avg_loss[e-1], avg_epoch_loss, base_smoothing)
            else:
                smoothing_factor = trainer.params['base_smoothing_rate']

        avg_loss.append(avg_epoch_loss)  # Store epoch loss for next iteration
        f = open('loss_file.txt', 'a+')
        f.write("Smoothing factor taken as: " + ' ' + str(trainer.params['label_smoothness']) + '\n'+
                "Average loss in this epoch" + ' ' + str(avg_epoch_loss) + '\n'
                + "Total loss so far is:" + ' ' + str(avg_loss) + '\n')
        f.close()
        print(f"Epoch {e}: Loss = {avg_epoch_loss:.4f}, Updated Smoothing Factor = {smoothing_factor:.4f}")

        print("Start evaluation after epoch: " + str(e))
        trainer.model.eval()
        index,results = evaluator.evaluate(trainer.model)
        index, mrr = evaluator.evaluate_mrr(trainer.model)
        print("---------------------------Results in Epoch------------------------:" + str(e))
        print(results)
        #Recall writing in a file
        f = open(trainer.params["training_result_update_file"], 'a+')
        f.write("Results in Epoch: " + str(e) + ' ' + str(results) + '\n')
        f.close()
        #writing mrrs
        f1 = open('Results_Mrr.txt', 'a+')
        f1.write("Results in Epoch: " + str(e) + ' ' + str(mrr) + '\n')
        f1.close()
        encoding_map = encode_documents(documents, trainer.model, trainer.collator)
        epoch_output_folder_path = os.path.join(
            trainer.params["model_dump_folder"], "epoch_{}".format(e)
        )
        #save_model(trainer.model,trainer.tokenizer,  epoch_output_folder_path)
        trainer.model.train()


    return final_output


'''
def train(epochs):
    trainer, evaluator, train_dataloader, optimizer, scheduler, entities, documents, doc_to_ent = load_train_blink_Ranking_Model()
    trainer.model.train()
    index, results = evaluator.evaluate(trainer.model)
    index, mrr = evaluator.evaluate_mrr(trainer.model)
    print(results)
    print(mrr)
    encoding_map = encode_documents(documents, trainer.model, trainer.collator)

    smoothing_factor = 0.1  # Label smoothing factor

    for e in range(epochs):
        num_batch = 0
        iter_ = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(iter_):
            # Create batch and apply label smoothing
            batch = data_processing.create_batch_index(batch[0], entities, list(entities[batch[0]]), encoding_map,
                                                       index, doc_to_ent)

            # Smooth the labels in the batch
            if "labels" in batch:
                batch["labels"] = apply_label_smoothing(batch["labels"], smoothing_factor)

            # Forward pass
            logits, loss = trainer.make_forward_pass(batch, step)

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
            if num_batch % trainer.evaluate_after == 0:
                print("Start evaluation in epoch:" + str(e) + " batch: " + str(num_batch))
                trainer.model.eval()
                index, results = evaluator.evaluate(trainer.model)
                index, mrr = evaluator.evaluate_mrr(trainer.model)
                print(results)
                print(mrr)
                encoding_map = encode_documents(documents, trainer.model, trainer.collator)
                trainer.model.train()

        print("Start evaluation after epoch: " + str(e))
        trainer.model.eval()
        index, results = evaluator.evaluate(trainer.model)
        index, mrr = evaluator.evaluate_mrr(trainer.model)
        print("---------------------------Results in Epoch------------------------:" + str(e))
        print(results)
        with open(trainer.params["training_result_update_file"], 'a+') as f:
            f.write("Results in Epoch: " + str(e) + str(results) + '\n')
        with open('Results_Mrr.txt', 'a+') as f1:
            f1.write("Results in Epoch: " + str(e) + str(mrr) + '\n')
        encoding_map = encode_documents(documents, trainer.model, trainer.collator)
        epoch_output_folder_path = os.path.join(
            trainer.params["model_dump_folder"], "epoch_{}".format(e)
        )
        trainer.model.train()

'''



#train(10, 0.000001)