    import json
import pickle
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from tqdm import tqdm, trange
from parameters import RankingParser
from models.E5 import E5Ranker
# from models.qwen3 import Qwen3Ranker
# from models.llama3 import Llama3Ranker
# from models.Llama3_LBW import Llama3LBWRanker
from models.llama_decoder import LlamaDecoderRanker
from models.qwen3_decoder import Qwen3DecoderRanker
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


def load_train_blink_Ranking_Model():
    parser = RankingParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)
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

    # for Llama3 (NEW)
    # if params["found_model"] == "llama3":
    #     train_inst = Trainer.TrainerLlama3(params=params, evaluate_after_batch=params["eval_interval"], device=device)
    #     print("✓ Using Llama3Ranker for dense retrieval")

    # if params["found_model"]=="llama3_lbw":
    #     train_inst = Trainer.TrainerLlama3LBW(params=params, evaluate_after_batch=params["eval_interval"], device=device)

    # # Adapter lines
    # params['lora_adapter_path'] = 'Finetuning/finetuned_models/llama-3.2-1b-lora/final'


    if params["found_model"] == "llama_decoder":
        train_inst = Trainer.TrainerLlamaDecoder(
            params=params, 
            evaluate_after_batch=params["eval_interval"], 
            device=device
        )
        print("Using LlamaDecoderRanker (LBW) for dense retrieval")


    # for qwen3
    if params["found_model"]=="qwen3":
        train_inst = Trainer.TrainerQwen3(params=params, evaluate_after_batch=params["eval_interval"], device=device)

    # for qwen3_decoder (LBW)
    if params["found_model"] == "qwen3_decoder":
        train_inst = Trainer.TrainerQwen3Decoder(
            params=params,
            evaluate_after_batch=params["eval_interval"],
            device=device
        )
        print("Using Qwen3DecoderRanker (LBW) for dense retrieval")

    #for aida
    # if params["dataset"] == "aida":
    #     dp=Aida_joint_el()
    #     tk = BertTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
    #     entities, documents, doc_to_ent=dp.read_ds_to_list("data/aida/wikidata/aida_train",tk)
    #entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/aida_train",tk)

    if params["dataset"] == "aida":
        dp = Aida_joint_el()
        # Use Qwen3 tokenizer - model_id from params
        from transformers import AutoTokenizer
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/aida_splits/aida_train", tk)

    if params["dataset"] == "ace2004":
        dp = Aida_joint_el()
        # Use Qwen3 tokenizer - model_id from params
        from transformers import AutoTokenizer
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/ace2004_splits/ACE2004_train", tk)

    if params["dataset"] == "aquaint":
        dp = Aida_joint_el()
        # Use Qwen3 tokenizer - model_id from params
        from transformers import AutoTokenizer
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/AQUAINT_splits/AQUAINT_train", tk)

    if params["dataset"] == "iitb-fix":
        dp = Aida_joint_el()
        # Use Qwen3 tokenizer - model_id from params
        from transformers import AutoTokenizer
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/iitb-fix_splits/iitb-fix_train", tk)

    if params["dataset"] == "kore50":
        dp = Aida_joint_el()
        # Use Qwen3 tokenizer - model_id from params
        from transformers import AutoTokenizer
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/KORE50_splits/KORE50_train", tk)

    if params["dataset"] == "msnbc":
        dp = Aida_joint_el()
        # Use Qwen3 tokenizer - model_id from params
        from transformers import AutoTokenizer
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/MSNBC_splits/MSNBC_train", tk)

    if params["dataset"] == "n3reuters128":
        dp = Aida_joint_el()
        # Use Qwen3 tokenizer - model_id from params
        from transformers import AutoTokenizer
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/N3-Reuters-128_splits/N3-Reuters-128_train", tk)

    if params["dataset"] == "n3rss500":
        dp = Aida_joint_el()
        # Use Qwen3 tokenizer - model_id from params
        from transformers import AutoTokenizer
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/N3-RSS-500_splits/N3-RSS-500_train", tk)

    if params["dataset"] == "spotlight":
        dp = Aida_joint_el()
        # Use Qwen3 tokenizer - model_id from params
        from transformers import AutoTokenizer
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        entities, documents, doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/spotlight_splits/spotlight_train", tk)

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
        # Use Qwen3 tokenizer - model_id from params
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,filehandler=dp.read_ds_to_list,file="data/aida/wikidata/aida_splits/aida_testa",tokenizer=tk)

    if params["dataset"] == "ace2004":
        # Use Qwen3 tokenizer - model_id from params
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,filehandler=dp.read_ds_to_list,file="data/aida/wikidata/ace2004_splits/ACE2004_testa",tokenizer=tk)

    if params["dataset"] == "aquaint":
        # Use Qwen3 tokenizer - model_id from params
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,filehandler=dp.read_ds_to_list,file="data/aida/wikidata/AQUAINT_splits/AQUAINT_testa",tokenizer=tk)

    if params["dataset"] == "iitb-fix":
        # Use Qwen3 tokenizer - model_id from params
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,filehandler=dp.read_ds_to_list,file="data/aida/wikidata/iitb-fix_splits/iitb-fix_testa",tokenizer=tk)

    if params["dataset"] == "kore50":
        # Use Qwen3 tokenizer - model_id from params
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,filehandler=dp.read_ds_to_list,file="data/aida/wikidata/KORE50_splits/KORE50_testa",tokenizer=tk)

    if params["dataset"] == "msnbc":
        # Use Qwen3 tokenizer - model_id from params
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,filehandler=dp.read_ds_to_list,file="data/aida/wikidata/MSNBC_splits/MSNBC_testa",tokenizer=tk)

    if params["dataset"] == "n3reuters128":
        # Use Qwen3 tokenizer - model_id from params
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,filehandler=dp.read_ds_to_list,file="data/aida/wikidata/N3-Reuters-128_splits/N3-Reuters-128_testa",tokenizer=tk)

    if params["dataset"] == "n3rss500":
        # Use Qwen3 tokenizer - model_id from params
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,filehandler=dp.read_ds_to_list,file="data/aida/wikidata/N3-RSS-500_splits/N3-RSS-500_testa",tokenizer=tk)

    if params["dataset"] == "spotlight":
        # Use Qwen3 tokenizer - model_id from params
        model_id = params.get('model_id', 'meta-llama/Llama-3.2-3B')
        tk = AutoTokenizer.from_pretrained(model_id)
        evaluator_inst = Evaluator.IndexEvaluator(params=params, collator=train_inst.collator,filehandler=dp.read_ds_to_list,file="data/aida/wikidata/spotlight_splits/spotlight_testa",tokenizer=tk)

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
    data_loader = DataLoader(documents, shuffle=True, batch_size=10,
                             collate_fn=collator.collate_context)
    #iter_ = tqdm(data_loader, desc="Encode Train Documents")
    doc_encodings = []
    with torch.no_grad():
      for step, batch in enumerate(data_loader):
          if not isinstance(model,E5Ranker) and not isinstance (model, LlamaDecoderRanker) and not isinstance (model, Qwen3DecoderRanker):
              context_input = batch
              # candidate_input = batch["candidate_input"]
              # labels=[0 for i in range(batch["candidate_input"].size(0))]
              # label_input = batch[0]["label_idx"].to(device)
              # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
              # context_input, candidate_input, label_input = batch
              encodings = model.encode_context(context_input).tolist()
          else:
              encodings=model.encode_context(batch).tolist()
          doc_encodings.extend(encodings)
          if torch.cuda.is_available():
                torch.cuda.empty_cache()
    encoding_map={}
    for i in range(len(doc_encodings)):
        encoding_map[documents[i]]=doc_encodings[i]
    return encoding_map

def train():
    #trainer,evaluator, train_dataloader, optimizer, scheduler = load_train_only_Graph_Model(device)
    trainer, evaluator, train_dataloader, optimizer, scheduler,entities,documents,doc_to_ent = load_train_blink_Ranking_Model()
    # trainer.model.train()
    trainer.model.eval()
    epochs= trainer.params['num_epochs']
    #print(evaluator.evaluate(trainer.model))
    index,results, mrr =evaluator.evaluate(trainer.model)
    #index, mrr = evaluator.evaluate_mrr(trainer.model)
    print(f"Recall: {results:.5f}")
    #print(mrr)

    f = open(trainer.params["training_result_update_file"], 'a+')
    with open("val_ece_log.txt", "a+") as f_ece:
        f_ece.write("####Noise ratio taken as: " + ' ' + str(trainer.params['noise_ratio'])+ '\n')
    f.write("####Noise ratio taken as: " + ' ' + str(trainer.params['noise_ratio']) + '\n'
            + "##Relaxation factor taken as: " + ' ' + str(trainer.params['relaxation_param']) + '\n')
            #+"##Smoothing factor taken as: " + ' ' + str(trainer.params['label_smoothness']) + '\n')
    f.close()
    # writing mrrs
    f1 = open('Results_Mrr', 'a+')
    f1.write("####Noise ratio taken as: " + ' ' + str(trainer.params['noise_ratio']) + '\n'
             + "##Relaxation factor taken as: " + ' ' + str(trainer.params['relaxation_param']) + '\n')
            #+"##Smoothing factor taken as: " + ' ' + str(trainer.params['label_smoothness']) + '\n')
    f1.close()
    encoding_map = encode_documents(documents, trainer.model, trainer.collator)

    # Exit early if only evaluation is requested
    if trainer.params.get('evaluate', False):
        print("Evaluation complete. Exiting (--evaluate flag was set).")
        return

    base_smoothing = float(trainer.params['base_smoothing_rate'])
    avg_loss = []
    encoding_map=encode_documents(documents,trainer.model,trainer.collator)
    for e in range(epochs):
        all_logits = []
        all_labels = []
        num_batch = 0
        total_loss = 0.0
        final_output = 0
        avg_epoch_loss = 0.0

        # step=0
        iter_ = tqdm(train_dataloader, desc="Training")
        EPOCH_FILE = "epoch.txt"
        with open(EPOCH_FILE, "w") as f:
            f.write(str(e))
        for step, batch in enumerate(iter_):
            #batch=data_processing.create_batch_ent(batch[0],list(entities[batch[0]]),random.sample(list(documents),1000),doc_to_ent)
            #batch=data_processing.create_batch_index(batch[0],entities,list(entities[batch[0]]),encoding_map,index,doc_to_ent)
            #batch = data_processing.create_batch_index_document(batch[0], entities,  encoding_map,
            #                                           index, doc_to_ent)
            batch=data_processing.create_batch_label_noise(batch[0],entities,list(entities[batch[0]]),encoding_map,index,doc_to_ent,num_noise_labels=trainer.params["noise_ratio"])
            logits, loss = trainer.make_forward_pass(batch,step)

            # --- Begin ECE-safe padding logic ---
            curr_width = logits.shape[1]
            if len(all_logits) > 0:
                max_width = max(curr_width, max(log.shape[1] for log in all_logits))
            else:
                max_width = curr_width

            # Pad current logits if needed
            if logits.shape[1] < max_width:
                pad_width = max_width - logits.shape[1]
                logits = torch.nn.functional.pad(logits, (0, pad_width), value=float('-inf'))

            # Pad previous logits if needed
            for i in range(len(all_logits)):
                if all_logits[i].shape[1] < max_width:
                    pad_width = max_width - all_logits[i].shape[1]
                    all_logits[i] = torch.nn.functional.pad(all_logits[i], (0, pad_width), value=float('-inf'))

            all_logits.append(logits.cpu())
            all_labels.append(torch.tensor(batch[2]))

            # Adding for adaptive label smoothing
            total_loss = total_loss + loss.item()  # Adding up the loss
            if trainer.params['adaptive_label_smoothing'] == 'yes':
                if e == 0:
                    trainer.params["label_smoothness"] = base_smoothing
                else:
                    trainer.params["label_smoothness"] = smoothing_factor  # Use current smoothing factor


            if trainer.grad_acc_steps > 1:
                loss = loss / trainer.grad_acc_steps
            loss.backward()
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
                #print(evaluator.evaluate(trainer.model))
                #print(evaluator.evaluate_mrr(trainer.model))
                index,results, mrr = evaluator.evaluate(trainer.model)
                #index, mrr = evaluator.evaluate_mrr(trainer.model)
                print(results)
                #print(mrr)
                encoding_map = encode_documents(documents, trainer.model, trainer.collator)
                #epoch_output_folder_path = os.path.join(
                #    "ranker_gr", "epoch_{}_{}".format(e, num_batch))
                #save_model(model, model.tokenizer, epoch_output_folder_path)
                trainer.model.train()

        print("Start evaluation after epoch: " + str(e))
        trainer.model.eval()
        index,results, mrr = evaluator.evaluate(trainer.model)

        print("---------------------------Results in Epoch------------------------:" + str(e))
        print(results)
        # **Epoch-Level Loss Computation**
        avg_epoch_loss = total_loss / len(train_dataloader)
        # **Update smoothing factor AFTER the epoch completes**
        if trainer.params['adaptive_label_smoothing'] == 'yes':
            if avg_loss:  # Ensure there's a previous loss recorded
                smoothing_factor = adaptive_smoothing_loss(avg_loss[e - 1], avg_epoch_loss, base_smoothing)
            else:
                smoothing_factor = trainer.params['base_smoothing_rate']

        avg_loss.append(avg_epoch_loss)  # Store epoch loss for next iteration
        f = open('loss_file.txt', 'a+')
        f.write("Smoothing factor taken as: " + ' ' + str(trainer.params["label_smoothness"]) + '\n'
                + "Average loss in this epoch" + ' ' + str(avg_epoch_loss) + '\n'
                + "Average loss in the previous epoch:" + ' ' + str(avg_loss[e - 1]) + '\n'
                + "Total loss so far is:" + ' ' + str(avg_loss) + '\n')
        f.close()
        #print(f"Epoch {e}: Loss = {avg_epoch_loss:.4f}, Updated Smoothing Factor = {smoothing_factor:.4f}")
        # Recall writing in a file
        f = open(trainer.params["training_result_update_file"], 'a+')
        f.write("Results in Epoch " + str(e) + " : " + str(results) + '\n')
        f.close()
        # writing mrrs
        f1 = open('Results_Mrr.txt', 'a+')
        f1.write("Results in Epoch " + str(e) + " : " + str(mrr) + '\n')
        f1.close()

        # computing calibration error
        ece_metric = ECEMetric(n_bins=15)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        train_ece = ece_metric(all_logits, all_labels)

        print(f"*********** [Epoch {e}] Training ECE ************: {train_ece:.4f}")
        # Optionally log to file
        with open("train_ece_log.txt", "a+") as f_ece:
            f_ece.write(f"Epoch {e}, Training ECE = {train_ece:.4f}\n")

        train_recall_at_5 = compute_recall_at_k(all_logits, all_labels, k=5)
        print(f"*********** [Epoch {e}] Training Recall@5 ************: {train_recall_at_5:.4f}")

        # Optional: write to log file
        with open("train_recall_log.txt", "a+") as f_rec:
            f_rec.write(f"Epoch {e}, Training Recall@5 = {train_recall_at_5:.4f}\n")
        encoding_map = encode_documents(documents, trainer.model, trainer.collator)
        epoch_output_folder_path = os.path.join(
            trainer.params["model_dump_folder"], "epoch_{}".format(e)
        )

        trainer.model.train()
    #save_model(trainer.model, trainer.tokenizer, epoch_output_folder_path)
    return results
#train(10)


# Class for computing expected calibration error per epoch
class ECEMetric:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def __call__(self, logits, labels):
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)

        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)

        for i in range(self.n_bins):
            # Define bin range
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Mask for predictions in this bin
            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            num_in_bin = mask.sum().item()

            if num_in_bin > 0:
                accuracy_in_bin = accuracies[mask].float().mean()
                avg_confidence_in_bin = confidences[mask].mean()
                ece += (num_in_bin / len(logits)) * torch.abs(avg_confidence_in_bin - accuracy_in_bin)

        return ece.item()


def compute_recall_at_k(logits, labels, k=5):
    """
    Computes Recall@k over a batch of predictions.
    `logits`: Tensor of shape (batch_size, num_candidates)
    `labels`: Tensor of shape (batch_size,), containing the true index
    """
    # Get top-k predicted indices
    topk_preds = torch.topk(logits, k, dim=1).indices  # shape: (batch_size, k)

    # Expand labels to compare with top-k
    labels = labels.view(-1, 1).expand_as(topk_preds)  # shape: (batch_size, k)

    # Compare and compute recall
    correct = (topk_preds == labels).any(dim=1).float()  # shape: (batch_size,)
    recall_at_k = correct.mean().item()
    return recall_at_k

train()
