import Evaluator
from parameters import RankingParser
import torch
from collator import Biencoder_Collator,E5collator
from models.BiEncoder import BiEncoderRanker
from models.E5 import E5Ranker
import data_processing
from data_processing import Aida_joint_el
from transformers import AutoTokenizer
from pytorch_transformers.tokenization_bert import BertTokenizer

device = torch.device(
            "cuda:1" if torch.cuda.is_available() else "cpu")
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
    #params["path_to_model"]="ranker_aida/epoch_"+str(i)+"/pytorch_model.bin"
    model=BiEncoderRanker(params)
    tk = BertTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
    #for e5
    #model = E5Ranker()
    model.load_state_dict(torch.load("ranker_aida_e5/epoch_"+str(i)+"/pytorch_model.bin", weights_only=True))
    model.to(device)

    #for biencoder
    collator = Biencoder_Collator(tokenizer=model.tokenizer,args=params, device=device)
    #for e5
    #tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
    #collator = E5collator(tokenizer=tokenizer,device=device)
    #for aida

   #
    #for lcquad
    entities_train,dk,_=data_processing.process_lcquad_file("data/train/lcquad.json")
    evaluator = Evaluator.IndexEvaluator(params=params, collator=collator)
    # for mintaka
    #entities_train, dk, _ = data_processing.process_minitaka_file("data/mintaka/mintaka_train.json")
    #evaluator = Evaluator.IndexEvaluator(params=params, collator=collator,False)

    #for aida
    #dp = Aida_joint_el()
    #entities_train, dk, _ = dp.read_ds_to_list("data/aida/wikidata/aida_train", tk)
    #eval_entities, eval_documents, eval_doc_to_ent = dp.read_ds_to_list("data/aida/wikidata/aida_testa",
    #                                                                    tk)
    #evaluator = Evaluator.IndexEvaluator(params=params, collator=collator, entities=eval_entities
    #                                          , doc_to_ent=eval_doc_to_ent, documents=eval_documents)

    evaluator.entities.extend(entities_train.keys())
    _,results=evaluator.evaluate(model)
    _,mrr= evaluator.evaluate_mrr(model) #return mrr
    results_all.append(results)
    results_mrr.append(mrr) #return mrr
    print(f"epoch{i}: {results}")
    print(f"epoch{i}: {mrr}")
for i in range(10):
    print(f"epoch{i}: {results_all[i]}")
    print(f"epoch{i}: {results_mrr[i]}")