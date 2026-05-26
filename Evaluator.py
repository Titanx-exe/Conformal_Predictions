import random
import json
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm, trange
import data_processing
import indexing
import conformal
from models.E5 import E5Ranker
# from models.qwen3 import Qwen3Ranker
# from models.llama3 import Llama3Ranker
from models.llama_decoder import LlamaDecoderRanker
from models.qwen3_decoder import Qwen3DecoderRanker
from torchmetrics.retrieval import RetrievalMRR

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs == labels

class EvaluatorCrossEncoder:
    def __init__(self,data, params, candidate_size):
        self.entities,self.documents,self.doc_to_ent=data.process_lcquad_file("data/test/lcquad.json")
        self.candidate_size = candidate_size
        self.params = params



    def generate_random_samples(self):
        samples=[]
        for doc in self.documents:
            entities=self.doc_to_ent[doc]
            for ent in entities:
                canidates=[]
                samples.append((doc,ent))
                while len(canidates)>self.candidate_size:
                    cnd=random.choice(entities.keys())
                    if not cnd in entities:
                        canidates.append(cnd)
        return samples
    def evaluate(self,
        model,random_samples=True
    ):
        if not random_samples:
            samples=[]
        else:
            samples=self.generate_random_samples()
        data_loader=DataLoader(samples, shuffle=True, batch_size=1,
                                  )
        model.eval()
        if self.params["silent"]:
            iter_ = data_loader
        else:
            iter_ = tqdm(data_loader, desc="Evaluation")

        results = {}

        eval_accuracy = 0.0
        nb_eval_examples = 0
        nb_eval_steps = 0

        for step, batch in enumerate(iter_):
            candidate_input = batch["candidate_encodings"]

            # label_input = batch[0]["label_idx"].to(device)
            # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
            # context_input, candidate_input, label_input = batch
            with torch.no_grad():
                eval_loss, logits = model(candidate_input, label_input=batch["labels"], context_len=32)

                logits = logits.detach().cpu().numpy()
            # Using in-batch negatives, the label ids are diagonal
            label_ids = torch.LongTensor(
                    torch.arange(self.params["eval_batch_size"])
            ).numpy()
            tmp_eval_accuracy, _ = accuracy(logits, batch["labels"].cpu().numpy())

            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += candidate_input.size(0)
            nb_eval_steps += 1

        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        print("Eval accuracy: %.5f" % normalized_eval_accuracy)
        results["normalized_accuracy"] = normalized_eval_accuracy
        return results

class EvaluatorBiEncoder:
    def __init__(self, params, candidate_size,collator):
        #self.data_loader = data_loader
        self.candidate_size = candidate_size
        self.params = params
        self.collator=collator
        self.entities,self.documents,self.doc_to_ent=data_processing.process_lcquad_file("data/test/lcquad.json")


    def generate_random_samples(self):
        samples=[]
        for doc in self.documents:
            entities=self.doc_to_ent[doc]
            for ent in entities:
                candidates=[ent]
                while len(candidates)<self.candidate_size:
                    cnd=random.choice(list(self.entities.keys()))
                    if not cnd in entities:
                        candidates.append(cnd)
                samples.append((doc,candidates))
        return samples
    def evaluate(self,model,random_samples=True):
        if not random_samples:
            samples=[]
        else:
            samples=self.generate_random_samples()
        data_loader=DataLoader(samples, shuffle=True, batch_size=100,
                                  collate_fn=self.collator.collate_batch_eval)
        if self.params["silent"]:
            iter_ = data_loader
        else:
            iter_ = tqdm(data_loader, desc="Evaluation")

        results = {}

        eval_accuracy = 0.0
        nb_eval_examples = 0
        nb_eval_steps = 0

        for step, batch in enumerate(iter_):

            context_input = batch["context_input"]
            candidate_input = batch["candidate_input"]
            labels=[0 for i in range(batch["candidate_input"].size(0))]
            label_ids=torch.tensor(labels,device=model.device)
            #label_input = batch[0]["label_idx"].to(device)
            # label_input = torch.LongTensor(torch.zeros(candidate_input.size(0),dtype=torch.int64)).to(device)
            # context_input, candidate_input, label_input = batch
            with torch.no_grad():
                eval_loss, logits = model(context_input, candidate_input, label_input=label_ids)

            logits = logits.detach().cpu().numpy()
            # Using in-batch negatives, the label ids are diagonal
            #label_ids = torch.LongTensor(
            #torch.arange(self.params["eval_batch_size"])

            #).numpy()
            tmp_eval_accuracy, _ = accuracy(logits, label_ids.cpu().numpy())

            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += context_input.size(0)
            nb_eval_steps += 1

        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
        print("Eval accuracy: %.5f" % normalized_eval_accuracy)
        results["normalized_accuracy"] = normalized_eval_accuracy
        return results




class IndexEvaluator:

    def __init__(self, params,collator,filehandler,file,tokenizer=None):
        #self.data_loader = data_loader
        #self.candidate_size = candidate_size
        self.params = params
        self.collator=collator
        self.filehandler = filehandler
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.entities, self.documents, self.doc_to_ent = filehandler(file,tokenizer)
        else: self.entities, self.documents, self.doc_to_ent = filehandler(file)
        '''
        if use_lcquad:
            self.entities,self.documents,self.doc_to_ent=data_processing.process_lcquad_file("data/test/lcquad.json")
        else:
            self.entities, self.documents, self.doc_to_ent = data_processing.process_minitaka_file("data/mintaka/mintaka_test.json")
        '''
        self.documents=sorted(list(self.documents))
        self.entities=sorted(list(self.entities.keys()))

        # Conformal settings
        self.conformal_method = self.params.get("conformal_method", "minmax")
        self.conformal_epsilon = float(self.params.get("conformal_epsilon", 0.1))
        self.conformal_coverage = self.params.get("conformal_coverage", "entity")
        self.calibration_file = self.params.get("calibration_file", None)
        self.delta = float(self.params.get("conformal_delta", 1e-8))

        # State
        self.q_hat = None
    '''
    def __init__(self,params,collator, entities, documents, doc_to_ent):
        self.entities=entities
        self.collator = collator
        self.documents=documents
        self.doc_to_ent=doc_to_ent
        self.params = params
        self.documents = list(self.documents)
        self.entities = list(self.entities.keys())
    '''
    def calibrate(self, model, cal_file=None):
        """Compute q_hat on the calibration set."""
        return self._calibrate_with_index(model, indexing.index_entities(model, self.entities, self.collator), cal_file)

    def _encode_documents(self, model, documents, desc):
        data_loader=DataLoader(documents, shuffle=False, batch_size=50,
                                  collate_fn=self.collator.collate_context)
        if self.params["silent"]:
            iter_ = data_loader
        else:
            iter_ = tqdm(data_loader, desc=desc)
        doc_encodings=[]
        with torch.no_grad():
            for step, batch in enumerate(iter_):
                if not isinstance(model,E5Ranker) and not isinstance(model, LlamaDecoderRanker) and not isinstance(model, Qwen3DecoderRanker):
                    encodings=model.encode_context(batch).tolist()
                else:
                    encodings=model.encode_context(batch)
                doc_encodings.extend(encodings)
        return doc_encodings

    def _compute_nonconformity(self, scores):
        scores_arr = np.array(scores, dtype=np.float32)
        if self.conformal_method == "softmax":
            return conformal.softmax_nc(scores_arr)
        elif self.conformal_method == "minmax":
            return conformal.minmax_nc(scores_arr, delta=self.delta)
        elif self.conformal_method == "margin":
            return conformal.margin_nc(scores_arr)
        else:
            raise ValueError(f"Unknown conformal method: {self.conformal_method}")

    def _calibrate_with_index(self, model, index, cal_file=None):
        """Compute q_hat on the calibration set using an existing entity index."""
        cal_file = cal_file or self.calibration_file
        if cal_file is None:
            raise ValueError("No calibration_file provided for conformal calibration.")

        # Load calibration data
        if self.tokenizer is not None:
            _, cal_docs, cal_doc_to_ent = self.filehandler(cal_file, self.tokenizer)
        else:
            _, cal_docs, cal_doc_to_ent = self.filehandler(cal_file)

        cal_docs = sorted(list(cal_docs))

        K = int(self.params.get("conformal_K", 100))
        cal_doc_encodings = self._encode_documents(model, cal_docs, "Encode Calibration Queries")
        found_ents = index.search(cal_doc_encodings, K)
        found_scores = np.array(index.last_scores, copy=True)

        cal_nc_scores = []
        missing_gold_count = 0
        total_gold_count = 0

        for doc, candidates, scores in zip(cal_docs, found_ents, found_scores):
            golds = set(cal_doc_to_ent.get(doc, set()))
            total_gold_count += len(golds)

            nc_arr = self._compute_nonconformity(scores)
            doc_nc = []
            for ent, nc in zip(candidates, nc_arr):
                if ent in golds:
                    doc_nc.append(nc)

            missing_gold_count += (len(golds) - len(doc_nc))

            if not doc_nc:
                continue

            if self.conformal_coverage == "entity":
                cal_nc_scores.extend(doc_nc)
            elif self.conformal_coverage == "document":
                cal_nc_scores.append(max(doc_nc))
            else:
                raise ValueError(f"Unknown conformal coverage: {self.conformal_coverage}")

        self.q_hat = conformal.conformal_threshold(cal_nc_scores, self.conformal_epsilon)

        print(f"[Conformal] Calibrated on {len(cal_docs)} docs. "
              f"Pooled {len(cal_nc_scores)} NC scores. "
              f"Missing golds in candidate sets: {missing_gold_count}/{total_gold_count}. "
              f"Method={self.conformal_method}, Coverage={self.conformal_coverage}, "
              f"q_hat={self.q_hat:.5f}")

    def generate_random_samples(self):
        samples=[]
        for doc in self.documents:
            entities=self.doc_to_ent[doc]
            for ent in entities:
                candidates=[ent]
                while len(candidates)<self.candidate_size:
                    cnd=random.choice(list(self.entities.keys()))
                    if not cnd in entities:
                        candidates.append(cnd)
                samples.append((doc,candidates))
        return samples
    def evaluate(self,model,random_samples=True,k=10):
        k = int(self.params.get("top_k", k))
        self.entities = sorted(set(self.entities))
        index = indexing.index_entities(model,self.entities, self.collator)
        set_predictor = self.params.get("set_predictor", "topk")
        search_k = int(self.params.get("conformal_K", 100)) if set_predictor == "conformal" else k
        doc_encodings = self._encode_documents(model, self.documents, "Encode Eval Queries")
        found_ents=index.search(doc_encodings,search_k)
        eval_scores = np.array(index.last_scores, copy=True)
        raw_found_ents = [list(prediction) for prediction in found_ents]
        score_threshold = float(self.params.get("score_threshold", 0.8))
        print(f"Set predictor: {set_predictor}")
        all_index_scores = [scores for scores in eval_scores if len(scores) > 0]
        if all_index_scores:
            flat_scores = np.concatenate(all_index_scores)
            score_min = float(np.min(flat_scores))
            score_max = float(np.max(flat_scores))
            score_mean = float(np.mean(flat_scores))
            print(f"Inner product score range: min={score_min:.5f}, max={score_max:.5f}, mean={score_mean:.5f}")
        # --- Precompute all baseline results in one pass (skip if conformal) ---
        if set_predictor != "conformal":
            effective_k = self.params.get("top_k", k)
            topk_results = raw_found_ents

            # scoret with min-max normalization
            scoret_results = []
            scoret_sizes = []
            for prediction, scores in zip(found_ents, eval_scores):
                scores_arr = np.array(scores, dtype=np.float32)
                if len(scores_arr) > 0 and scores_arr.max() > scores_arr.min():
                    norm_scores = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
                else:
                    norm_scores = np.ones_like(scores_arr)
                filtered = [ent for ent, ns in zip(prediction, norm_scores) if ns >= score_threshold]
                scoret_results.append(filtered)
                scoret_sizes.append(len(filtered))

            # platt temperature-scaled mass
            npmp_epsilon = float(self.params.get("npmp_epsilon", 0.1))
            target_mass = 1.0 - npmp_epsilon
            temperature = float(self.params.get("platt_temperature", 1.0))
            npmp_results = []
            npmp_sizes = []
            for prediction, scores in zip(found_ents, eval_scores):
                probs = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)
                selected = []
                cumulative_mass = 0.0
                sorted_predictions = sorted(zip(prediction, probs), key=lambda item: float(item[1]), reverse=True)
                for ent, prob in sorted_predictions:
                    selected.append(ent)
                    cumulative_mass += float(prob)
                    if cumulative_mass >= target_mass:
                        break
                npmp_results.append(selected)
                npmp_sizes.append(len(selected))

            platt_results = []
            platt_sizes = []
            if temperature <= 0:
                raise ValueError("platt_temperature must be greater than 0")
            for prediction, scores in zip(found_ents, eval_scores):
                probs = torch.softmax(torch.tensor(scores, dtype=torch.float32) / temperature, dim=0)
                selected = []
                cumulative_mass = 0.0
                sorted_predictions = sorted(zip(prediction, probs), key=lambda item: float(item[1]), reverse=True)
                for ent, prob in sorted_predictions:
                    selected.append(ent)
                    cumulative_mass += float(prob)
                    if cumulative_mass >= target_mass:
                        break
                platt_results.append(selected)
                platt_sizes.append(len(selected))

        if set_predictor == "scoret":
            found_ents = scoret_results
            avg_set_size = sum(scoret_sizes) / len(scoret_sizes) if scoret_sizes else 0.0
            print(f"Score threshold: {score_threshold:.5f}")
            print(f"Average set size after threshold: {avg_set_size:.5f}")
            print(f"Min/Max set size after threshold: {min(scoret_sizes) if scoret_sizes else 0}/{max(scoret_sizes) if scoret_sizes else 0}")
        elif set_predictor == "npmp":
            found_ents=npmp_results
            avg_set_size=sum(npmp_sizes) / len(npmp_sizes) if npmp_sizes else 0.0
            print(f"NPMP epsilon: {npmp_epsilon:.5f}")
            print(f"NPMP target mass: {target_mass:.5f}")
            print(f"Average set size after NPMP: {avg_set_size:.5f}")
            print(f"Min/Max set size after NPMP: {min(npmp_sizes) if npmp_sizes else 0}/{max(npmp_sizes) if npmp_sizes else 0}")
        elif set_predictor == "platt":
            found_ents = platt_results
            avg_set_size = sum(platt_sizes) / len(platt_sizes) if platt_sizes else 0.0
            print(f"Platt epsilon: {npmp_epsilon:.5f}")
            print(f"Platt target mass: {target_mass:.5f}")
            print(f"Platt temperature: {temperature:.5f}")
            print(f"Average set size after Platt: {avg_set_size:.5f}")
            print(f"Min/Max set size after Platt: {min(platt_sizes) if platt_sizes else 0}/{max(platt_sizes) if platt_sizes else 0}")
        elif set_predictor == "conformal":
            if self.q_hat is None:
                self._calibrate_with_index(model, index, self.calibration_file)

            conformal_found_ents = []
            conformal_sizes = []
            for candidates, scores in zip(found_ents, eval_scores):
                nc_arr = self._compute_nonconformity(scores)
                selected = [ent for ent, nc in zip(candidates, nc_arr) if nc <= self.q_hat]
                conformal_found_ents.append(selected)
                conformal_sizes.append(len(selected))
            index.last_scores = eval_scores

            found_ents = conformal_found_ents
            avg_set_size = sum(conformal_sizes) / len(conformal_sizes) if conformal_sizes else 0.0
            print(f"Conformal method: {self.conformal_method}, q_hat: {self.q_hat:.5f}")
            print(f"Average set size after conformal filtering: {avg_set_size:.5f}")
            print(f"Min/Max set size: {min(conformal_sizes) if conformal_sizes else 0}/{max(conformal_sizes) if conformal_sizes else 0}")
        # Log per-document results
        if set_predictor == "conformal":
            self._log_conformal_results(conformal_found_ents, conformal_sizes)
        else:
            self._log_baseline_results(topk_results, scoret_results, npmp_results, platt_results)

        mrr = self.evaluate_mrr(raw_found_ents)
        all_labels=[]
        all_scores=[]
        indexes=[]
        all_found=0
        all_not_found=0
        total_eval_documents = len(self.documents)
        covered_documents = 0
        gold_coverage_sum = 0.0
        print("Total evaluation documents:" + str(total_eval_documents))
        for i in range(total_eval_documents):
            correct_entities=set(self.doc_to_ent[self.documents[i]])
            prediction=set(found_ents[i])
            if not prediction.isdisjoint(correct_entities):
                covered_documents+=1
            if len(correct_entities) > 0:
                gold_coverage_sum += len(correct_entities.intersection(prediction)) / len(correct_entities)
            for en in correct_entities:
                if en in prediction:
                    all_found+=1
                else:
                    all_not_found+=1
        print("found:"+str(all_found)+" not found:"+str(all_not_found))
        results=all_found/(all_not_found+all_found)
        empirical_coverage = covered_documents / total_eval_documents if total_eval_documents > 0 else 0.0
        gold_coverage = gold_coverage_sum / total_eval_documents if total_eval_documents > 0 else 0.0
        print(f"Empirical Coverage (Cov_cg): {empirical_coverage:.5f}")
        print(f"Gold Coverage: {gold_coverage:.5f}")
        #print(result)

        #Adding code for ece computation

        # ---------------------- Compute ECE ----------------------
        ece_metric = ECEMetric(n_bins=15)
        confidences = []
        labels = []

        for i in range(len(self.documents)):
            correct_entities = set(self.doc_to_ent[self.documents[i]])
            prediction = found_ents[i]
            if not prediction or i >= len(eval_scores) or len(eval_scores[i]) == 0:
                continue
            top_pred = prediction[0]  # Top-1 entity

            # If you have similarity scores returned by index.search()
            # Make sure to store them in a variable like: index.last_scores[i]
            # You may need to modify indexing.search() to return scores.
            confidence = eval_scores[i][0]  # top-1 score
            confidences.append(confidence)

            label = 1 if top_pred in correct_entities else 0
            labels.append(label)

        logits = torch.tensor(confidences).unsqueeze(1)  # shape (N, 1)
        labels = torch.tensor(labels)
        logits = torch.cat([logits, 1 - logits], dim=1)  # fake 2-class logits

        ece_value = ece_metric(logits, labels)
        print(f"**************************** Validation ECE: {ece_value:.5f}")
        with open("epoch.txt", "r") as f:
            data = f.read()  # Reads entire content as a string
            e = data

        with open("val_ece_log.txt", "a+") as f_ece:
            f_ece.write(f"Epoch {e}, Validation ECE = {ece_value:.4f}\n")
        # ---------------------------------------------------------

        # --- Baseline summary for all three predictors (skip if conformal) ---
        if set_predictor != "conformal":
            topk_rec, topk_emp, topk_gold = self._compute_recall_and_coverage(topk_results)
            scoret_rec, scoret_emp, scoret_gold = self._compute_recall_and_coverage(scoret_results)
            npmp_rec, npmp_emp, npmp_gold = self._compute_recall_and_coverage(npmp_results)
            platt_rec, platt_emp, platt_gold = self._compute_recall_and_coverage(platt_results)
            topk_mrr = self._compute_mrr(topk_results)
            scoret_mrr = self._compute_mrr(scoret_results)
            npmp_mrr = self._compute_mrr(npmp_results)
            platt_mrr = self._compute_mrr(platt_results)

            print("================ BASELINE RESULTS ================")
            print(f"[topk]   MRR: {topk_mrr:.5f}, Recall: {topk_rec:.5f}, EmpCov: {topk_emp:.5f}, GoldCov: {topk_gold:.5f}")
            print(f"[scoret] MRR: {scoret_mrr:.5f}, Recall: {scoret_rec:.5f}, EmpCov: {scoret_emp:.5f}, GoldCov: {scoret_gold:.5f}")
            print(f"[npmp]   MRR: {npmp_mrr:.5f}, Recall: {npmp_rec:.5f}, EmpCov: {npmp_emp:.5f}, GoldCov: {npmp_gold:.5f}")
            print(f"[platt]  MRR: {platt_mrr:.5f}, Recall: {platt_rec:.5f}, EmpCov: {platt_emp:.5f}, GoldCov: {platt_gold:.5f}")
            print("==================================================")
        else:
            conformal_rec, conformal_emp, conformal_gold = self._compute_recall_and_coverage(conformal_found_ents)
            conformal_mrr = self._compute_mrr(conformal_found_ents)
            avg_set_size = sum(conformal_sizes) / len(conformal_sizes) if conformal_sizes else 0.0
            print("================ CONFORMAL RESULTS ================")
            print(f"Method: {self.conformal_method}, epsilon: {self.conformal_epsilon}, q_hat: {self.q_hat:.5f}")
            print(f"MRR: {conformal_mrr:.5f}, Recall: {conformal_rec:.5f}, EmpCov: {conformal_emp:.5f}, GoldCov: {conformal_gold:.5f}")
            print(f"AvgSetSize: {avg_set_size:.5f}, Min: {min(conformal_sizes) if conformal_sizes else 0}, Max: {max(conformal_sizes) if conformal_sizes else 0}")
            print("===================================================")

        return index,results, mrr

    def evaluate_mrr(self, found_ents, random_samples=True, k=10):
        '''
        index = indexing.index_entities(model, self.entities, self.collator)

        data_loader = DataLoader(self.documents, shuffle=False, batch_size=100,
                                 collate_fn=self.collator.collate_context)

        if self.params["silent"]:
            iter_ = data_loader
        else:
            iter_ = tqdm(data_loader, desc="Encode Eval Queries")

        doc_encodings = []
        for step, batch in enumerate(iter_):
            if not isinstance(model, E5Ranker):
                context_input = batch
                encodings = model.encode_context(context_input).tolist()
            else:
                encodings = model.encode_context(batch)
            doc_encodings.extend(encodings)

        found_ents = index.search(doc_encodings, k)
        '''
        all_rr = []  # List to store reciprocal ranks for MRR calculation

        for i in range(len(self.documents)):
            correct_entities = set(self.doc_to_ent[self.documents[i]])  # Set of correct entities
            prediction = found_ents[i]  # Predicted entities

            # Find the rank of the first correct entity
            reciprocal_rank = 0
            for rank, entity in enumerate(prediction, start=1):
                if entity in correct_entities:
                    reciprocal_rank = 1 / rank
                    break

            all_rr.append(reciprocal_rank)

        mrr = sum(all_rr) / len(all_rr) if all_rr else 0
        print(f"Mean Reciprocal Rank (MRR): {mrr:.5f}")

        return mrr

    # --- Baseline logging and helper methods ---
    def _log_baseline_results(self, topk_ents, scoret_ents, npmp_ents, platt_ents):
        import os
        found_model = self.params.get("found_model", "unknown")
        if found_model in ["biencoder", "e5"]:
            model_name = found_model
        else:
            model_id = self.params.get("model_id", "unknown")
            model_name = os.path.basename(model_id) if model_id else "unknown"
        lora_adapter_path = self.params.get("lora_adapter_path", None)
        lora_epoch = self.params.get("lora_epoch", "final")
        results_dir = self.params.get("results_dir", None)
        if lora_adapter_path:
            lora_name = os.path.basename(lora_adapter_path.rstrip("/"))
            log_dir = os.path.join("baseline_results", model_name, f"{lora_name}_epoch_{lora_epoch}")
        else:
            log_dir = os.path.join("baseline_results", model_name)
        if results_dir is not None:
            log_dir = os.path.join(results_dir, log_dir)
        os.makedirs(log_dir, exist_ok=True)
        dataset = self.params.get("dataset", "unknown")
        arch = self.params.get("lbw_architecture", "NA")
        num_bidir = self.params.get("num_bidir_layers", "0")
        num_unsink = self.params.get("num_unsink_layers", "0")
        mask_type = self.params.get("mask_type", "NA")
        
        found_model = self.params.get("found_model", "unknown")
        if found_model in ["biencoder", "e5"]:
            filename = os.path.join(log_dir, f"{dataset}_baseline.json")
        else:
            filename = os.path.join(
                log_dir,
                f"{dataset}_{arch}_{num_bidir}_{num_unsink}_{mask_type}_baseline.json"
            )
            
        log_data = []
        for i in range(len(self.documents)):
            doc = self.documents[i]
            gold = set(self.doc_to_ent.get(doc, set()))
            config_dict = {"dataset": dataset}
            if found_model not in ["biencoder", "e5"]:
                config_dict.update({
                    "lbw_architecture": arch,
                    "num_bidir_layers": num_bidir,
                    "num_unsink_layers": num_unsink,
                    "mask_type": mask_type,
                })
            log_data.append({
                "config": config_dict,
                "document": doc,
                "gold_entities": sorted(list(gold)),
                "topk_entities": topk_ents[i],
                "scoret_entities": scoret_ents[i],
                "npmp_entities": npmp_ents[i],
                "platt_entities": platt_ents[i],
            })
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"[BaselineLog] {filename}")

    def _log_conformal_results(self, conformal_ents, conformal_sizes):
        import os
        found_model = self.params.get("found_model", "unknown")
        if found_model in ["biencoder", "e5"]:
            model_name = found_model
        else:
            model_id = self.params.get("model_id", "unknown")
            model_name = os.path.basename(model_id) if model_id else "unknown"
        lora_adapter_path = self.params.get("lora_adapter_path", None)
        lora_epoch = self.params.get("lora_epoch", "final")
        results_dir = self.params.get("results_dir", None)
        if lora_adapter_path:
            lora_name = os.path.basename(lora_adapter_path.rstrip("/"))
            log_dir = os.path.join("conformal_results", model_name, f"{lora_name}_epoch_{lora_epoch}")
        else:
            log_dir = os.path.join("conformal_results", model_name)
        if results_dir is not None:
            log_dir = os.path.join(results_dir, log_dir)
        os.makedirs(log_dir, exist_ok=True)
        dataset = self.params.get("dataset", "unknown")
        arch = self.params.get("lbw_architecture", "NA")
        num_bidir = self.params.get("num_bidir_layers", "0")
        num_unsink = self.params.get("num_unsink_layers", "0")
        mask_type = self.params.get("mask_type", "NA")
        
        found_model = self.params.get("found_model", "unknown")
        if found_model in ["biencoder", "e5"]:
            filename = os.path.join(log_dir, f"{dataset}_conformal.json")
        else:
            filename = os.path.join(
                log_dir,
                f"{dataset}_{arch}_{num_bidir}_{num_unsink}_{mask_type}_conformal.json"
            )
            
        log_data = []
        log_count = min(len(self.documents), len(conformal_ents), len(conformal_sizes))
        if log_count != len(self.documents):
            print(f"[ConformalLog] Warning: logging {log_count}/{len(self.documents)} documents because prediction count differs.")
        for i in range(log_count):
            doc = self.documents[i]
            gold = set(self.doc_to_ent.get(doc, set()))
            config_dict = {
                "dataset": dataset,
                "conformal_method": self.conformal_method,
                "conformal_epsilon": self.conformal_epsilon,
                "conformal_coverage": self.conformal_coverage,
            }
            if found_model not in ["biencoder", "e5"]:
                config_dict.update({
                    "lbw_architecture": arch,
                    "num_bidir_layers": num_bidir,
                    "num_unsink_layers": num_unsink,
                    "mask_type": mask_type,
                })
            log_data.append({
                "config": config_dict,
                "document": doc,
                "gold_entities": sorted(list(gold)),
                "conformal_entities": conformal_ents[i],
                "set_size": conformal_sizes[i],
            })
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"[ConformalLog] {filename}")

    def _compute_recall_and_coverage(self, predictions):
        all_found = 0
        all_not_found = 0
        total = len(self.documents)
        covered = 0
        gold_sum = 0.0
        if len(predictions) != total:
            print(f"[Metrics] Warning: {len(predictions)} prediction sets for {total} documents; scoring matched prefix only.")
        for i in range(min(total, len(predictions))):
            correct = set(self.doc_to_ent[self.documents[i]])
            pred = set(predictions[i])
            if not pred.isdisjoint(correct):
                covered += 1
            if len(correct) > 0:
                gold_sum += len(correct.intersection(pred)) / len(correct)
            for en in correct:
                if en in pred:
                    all_found += 1
                else:
                    all_not_found += 1
        recall = all_found / (all_found + all_not_found) if (all_found + all_not_found) > 0 else 0.0
        emp = covered / total if total > 0 else 0.0
        gold = gold_sum / total if total > 0 else 0.0
        return recall, emp, gold

    def _compute_mrr(self, found_ents):
        all_rr = []
        if len(found_ents) != len(self.documents):
            print(f"[MRR] Warning: {len(found_ents)} prediction sets for {len(self.documents)} documents; scoring matched prefix only.")
        for i in range(min(len(self.documents), len(found_ents))):
            correct = set(self.doc_to_ent[self.documents[i]])
            prediction = found_ents[i]
            rr = 0.0
            for rank, entity in enumerate(prediction, start=1):
                if entity in correct:
                    rr = 1.0 / rank
                    break
            all_rr.append(rr)
        return sum(all_rr) / len(all_rr) if all_rr else 0.0


class ECEMetric:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def __call__(self, logits, labels):
        """
        Computes Expected Calibration Error (ECE).
        Args:
            logits: Tensor of shape (N, C) where C is the number of classes.
            labels: Tensor of shape (N,) with integer class labels.
        Returns:
            Scalar ECE value.
        """
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)

        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            num_in_bin = mask.sum().item()

            if num_in_bin > 0:
                accuracy_in_bin = accuracies[mask].float().mean()
                avg_confidence_in_bin = confidences[mask].mean()
                ece += (num_in_bin / logits.size(0)) * torch.abs(avg_confidence_in_bin - accuracy_in_bin)

        return ece.item()