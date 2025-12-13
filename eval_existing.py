import Evaluator
from parameters import RankingParser
import torch
from collator import Biencoder_Collator,E5collator,Qwen3Collator,Llama3Collator,Llama3LBWCollator
from models.BiEncoder import BiEncoderRanker
from models.E5 import E5Ranker
from models.qwen3 import Qwen3Ranker
from models.llama3 import Llama3Ranker
from models.Llama3_LBW import Llama3LBWRanker
import data_processing
from data_processing import Aida_joint_el
from transformers import AutoTokenizer
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import random





# ---------- ECE utilities ----------
def compute_ece(logits, labels, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1, device=logits.device)

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        prop_in_bin = mask.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[mask].float().mean()
            conf_in_bin = confidences[mask].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return ece.item()

def compute_model_ece_over_eval(model, entities, collator, device, max_candidates=10):
    model.eval()
    all_logits = []
    all_labels = []

    dataset = list(entities.keys())
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator.collate_context)

    max_width = 0
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval ECE"):
            #for e5
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model.encode(inputs)
            if isinstance(model, Qwen3Ranker):
                # Qwen3 uses last_token_pool + normalization
                embeddings = model.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            elif isinstance(model, E5Ranker):
                # E5 uses average_pool
                embeddings = model.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            elif isinstance(model, Llama3Ranker):
                # Llama3 uses weighted average pooling
                embeddings = model.weighted_average_pool(outputs.hidden_states[-1], inputs['attention_mask'])
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")


            #ctx_embed, doc_embed = torch.split(embeddings, embeddings.size(0) // 2)
            midpoint = embeddings.size(0) // 2
            ctx_embed = embeddings[:midpoint]
            doc_embed = embeddings[midpoint:]

            logits = ctx_embed @ doc_embed.T  # shape: [B, N]
            labels = torch.arange(logits.size(0), device=device)

            logits_list.append(logits)
            labels_list.append(labels)

            # Track the max width (N) for padding
            if logits.shape[1] > max_width:
                max_width = logits.shape[1]

    # Pad logits so all are shape [B, max_width]
    padded_logits = []
    padded_labels = []

    for logits, labels in zip(logits_list, labels_list):
        pad_width = max_width - logits.shape[1]
        if pad_width > 0:
            pad = torch.full((logits.shape[0], pad_width), fill_value=-1e9, device=logits.device)
            logits = torch.cat([logits, pad], dim=1)
        padded_logits.append(logits)
        padded_labels.append(labels)

    logits_tensor = torch.cat(padded_logits, dim=0)
    labels_tensor = torch.cat(padded_labels, dim=0)
    return compute_ece(logits_tensor, labels_tensor)





device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
parser = RankingParser(add_model_args=True)
parser.add_training_args()
parser.add_eval_args()
args = parser.parse_args()
print(args)
params = args.__dict__
results_all=[]
results_mrr=[] #mrr AS
for i in range(10):
    #for biencoder
    #params["path_to_model"]="model/epoch_"+str(i)+"/pytorch_model.bin"
    #model=BiEncoderRanker(params)
    ## tk = BertTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
    #for e5
    ## model = E5Ranker()
    ## model.load_state_dict(torch.load("model/epoch_"+str(i)+"/pytorch_model.bin", weights_only=True))
    ## model.to(device)

    #for biencoder
    #collator = Biencoder_Collator(tokenizer=model.tokenizer,args=params, device=device)
    #for e5
    #tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
    #for BERT
    ## tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    ## collator = E5collator(tokenizer=tokenizer,device=device)
    #for aida

   #
    #for lcquad
    #entities_train,dk,_=data_processing.process_lcquad_file("data/train/lcquad.json")
    #evaluator = Evaluator.IndexEvaluator(params=params, collator=collator)
    # for mintaka
    if params["found_model"] == "e5":
        model = E5Ranker(device=device, params=params)
        model.load_state_dict(torch.load("model/epoch_"+str(i)+"/pytorch_model.bin", weights_only=True))
        model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        collator = E5collator(tokenizer=tokenizer, device=device)
    
    # for Qwen3 (ADD THIS)
    elif params["found_model"] == "qwen3":
        model = Qwen3Ranker(device=device, params=params)
        model.load_state_dict(torch.load("model/epoch_"+str(i)+"/pytorch_model.bin", weights_only=True))
        model.to(device)
        
        # Use tokenizer from model (already configured with left padding)
        tokenizer = model.tokenizer
        collator = Qwen3Collator(tokenizer=tokenizer, device=device)
    elif params["found_model"] == "llama3":
        model = Llama3Ranker(device=device, params=params)
        model.load_state_dict(torch.load("model/epoch_"+str(i)+"/pytorch_model.bin", weights_only=True))
        model.to(device)
        tokenizer = model.tokenizer
        collator = Llama3Collator(tokenizer=tokenizer, device=device)

    elif params["found_model"] == "llama3_lbw":
        from models.Llama3_LBW import Llama3LBWRanker
        from collator import Llama3LBWCollator
        model = Llama3LBWRanker(device=device, params=params)
        model.load_state_dict(torch.load("model/epoch_"+str(i)+"/pytorch_model.bin", weights_only=True))
        model.to(device)
        tokenizer = model.tokenizer
        collator = Llama3LBWCollator(tokenizer=tokenizer, device=device)
    
    else:
        raise ValueError(f"Unsupported model type: {params['found_model']}")
    entities_train, dk, _ = data_processing.process_minitaka_file("data/mintaka/mintaka_train.json")
    evaluator = Evaluator.IndexEvaluator(params=params, collator=collator,filehandler=data_processing.process_minitaka_file, file="data/mintaka/mintaka_test.json")

    #for aida
    #dp = Aida_joint_el()
    #entities_train, dk, _ = dp.read_ds_to_list("data/aida/wikidata/aida_train", tk)
    #eval_entities, eval_documents, eval_doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/aida_testa",
    #                                                                    tokenizer)
    #tk = BertTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
    #evaluator = Evaluator.IndexEvaluator(params=params, collator=collator,
    #                                          filehandler=dp.read_ds_to_list, file="data/aida/wikidata/aida_testa",
    #                                          tokenizer=tk)


    evaluator.entities.extend(entities_train.keys())
    _,results, mrr=evaluator.evaluate(model)
   # _,mrr= evaluator.evaluate_mrr(model) #return mrr
    #for e5
    ece = compute_model_ece_over_eval(model, entities_train, collator, device)

    results_all.append(results)
    results_mrr.append(mrr) #return mrr
    print(f"epoch{i}: {results}")
    print(f"epoch{i}: {mrr}")
    print(f"epoch{i}: ECE = {ece:.7f}")
for i in range(5):
    print(f"epoch{i}: {results_all[i]}")
    print(f"epoch{i}: {results_mrr[i]}")
