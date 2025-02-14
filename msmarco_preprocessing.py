import gzip
import pickle
from indexing import index_data
import torch
import random
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
all_relevant=set()
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
                all_relevant.add(l[2])
            else:
                queries[l[0]]["relevant"].append(l[2])
    return queries
#docs=pickle.load(open("data/msmarco/doc_dictionary","rb"))
queries=load_queries("data/msmarco/msmarco-docdev-queries.tsv.gz","data/msmarco/msmarco-docdev-qrels.tsv.gz")
#pickle.dump(queries,open("data/msmarco/eval_queries","wb"))


all_gold_documents={}
current_documents={}
count=0
with gzip.open("data/msmarco/msmarco-docs.tsv.gz", 'rt', encoding='utf8') as f:
    for line in f:
        l=line.split("\t")
        if l[0]in all_relevant:
            all_gold_documents[l[0]]=[l[1],l[2],l[3]]
        else:
            current_documents[l[0]]=[l[1],l[2],l[3]]
        if len(current_documents)==1000000:
            negatives=random.sample(list(current_documents.keys()),40000)
            for el in negatives:
                all_gold_documents[el]=current_documents[el]
            #pickle.dump(current_documents, open("data/msmarco/doc_dictionary_split_"+str(count), "wb"))
            current_documents={}
            count+=1
#pickle.dump(current_documents,open("data/msmarco/doc_dictionary_split_"+str(count),"wb"))
pickle.dump(all_gold_documents,open("data/msmarco/eval_documents_100000","wb"))
print("finished_preprocessing")

#index,idToIndex=index_data(model,list(queries.keys()),collator.collate_context,model.encode_context)
#print("finished_indexing")