import gzip
import pickle
from indexing import index_entities
import torch
from parameters import RankingParser
from models.BiEncoder import BiEncoderRanker
from pytorch_transformers.tokenization_bert import BertTokenizer
from collator import Biencoder_Collator
parser = RankingParser(add_model_args=True)
parser.add_training_args()
parser.add_eval_args()
args = parser.parse_args()
print(args)
params = args.__dict__
device = torch.device(
            "cuda:1" if torch.cuda.is_available() else "cpu")

model=BiEncoderRanker(params)
tk = BertTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
model.to(device)

#for biencoder
collator = Biencoder_Collator(tokenizer=model.tokenizer,args=params, device=device)


'''
doc_dictionary={}
with gzip.open("data/msmarco/msmarco-docs.tsv.gz", 'rt', encoding='utf8') as f:
    for line in f:
        l=line.split("\t")
        doc_dictionary[l[0]]=[l[1],l[2],l[3]]
pickle.dump(doc_dictionary,open("data/msmarco/doc_dictionary","wb"))
'''
def load_queries(filename_queries,filename_relevant):
    queries={}
    with gzip.open(filename_queries, 'rt', encoding='utf8') as f:
        for line in f:
            l = line.split("\t")
            queries[l[0]]={"text":l[1]}
    qr=set()
    with gzip.open(filename_relevant, 'rt', encoding='utf8') as f:
        for line in f:
            l = line.split(" ")
            if not "relevant" in queries[l[0]]:
                queries[l[0]]["relevant"]=[l[2]]
            else:
                queries[l[0]]["relevant"].append(l[2])
    return queries

queries=load_queries("data/msmarco/msmarco-doctrain-queries.tsv.gz","data/msmarco/msmarco-doctrain-qrels.tsv.gz")
index=index_entities(model,list(queries.keys())[:100],collator)
print("finished_indexing")