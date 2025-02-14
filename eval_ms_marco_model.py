import gzip
import pickle
from indexing import index_data
import torch
from tqdm import tqdm
import random
import faiss
from indexing import DenseFlatIndexer
import numpy
from indexing import generateVectors
from parameters import RankingParser
from models.BiEncoderHuggingface import BiEncoderRanker
from models.E5 import E5Ranker
import pickle
from transformers import AutoTokenizer
parser = RankingParser(add_model_args=True)
parser.add_training_args()
parser.add_eval_args()

# args = argparse.Namespace(**params)
args = parser.parse_args()
print(args)
params = args.__dict__

current_documents=[]
batch_size=40
device="cuda:0"
encodings=[]
doc_indices={}
model=E5Ranker(device=device)
#model = BiEncoderRanker(params,device=device)
#tokenizer=model.tokenizer
tokenizer=AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
model.load_state_dict(torch.load("ms_marco_models/e5/gausian/pytorch_model.bin", weights_only=True))
model.to(device)
#model.load_state_dict(torch.load("ms_marco_models/biencoder/no_noise/pytorch_model.bin", weights_only=True))
index=0
documents=pickle.load(open("data/msmarco/eval_documents_100000", "rb"))
docs=tqdm(list(documents.keys()))
eval_queries=pickle.load(open("data/msmarco/eval_queries", "rb"))
#with gzip.open("data/msmarco/msmarco-docs.tsv.gz", 'rt', encoding='utf8') as f:
query_encodings=[]
questions=["query: "+eval_queries[question]["text"] for question in eval_queries.keys()]
relevants=[eval_queries[question]["relevant"][0] for question in eval_queries.keys()]

for batch in range(0, len(eval_queries), batch_size):
    curr_batch = tokenizer(questions[batch:batch+batch_size], max_length=512, padding=True, truncation=True, return_tensors='pt')
    curr_batch.to(device)
    encs = model.encode_context(curr_batch).tolist()
    query_encodings.extend(encs)

for key in docs:
        #l=line.split("\t")
        current_documents.append((documents[key][1],documents[key][2]))
        doc_indices[index]=key
        index=index+1
        if len(current_documents)==batch_size:

            #candidates = ["title: " + cand[0] + "[SEP] context: " + cand[1] for
            #              cand in current_documents]
            candidates = ["passage: title: " + cand[0] + "[SEP] context: " + cand[1] for
                          cand in current_documents]
            batch = tokenizer(candidates, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch.to(device)
            current_documents=[]
            encs = model.encode_candidate(batch).tolist()
            encodings.extend(encs)
#candidates = ["title: " + cand[0] + "[SEP] context: " + cand[1] for
            #              cand in current_documents]
candidates = ["passage: title: " + cand[0] + "[SEP] context: " + cand[1] for
                          cand in current_documents]
batch = tokenizer(candidates, max_length=512, padding=True, truncation=True, return_tensors='pt')
batch.to(device)
encs = model.encode_candidate(batch).tolist()
encodings.extend(encs)

print("start geneationg vectors")
x_dim = len(encodings)
y_dim = len(encodings[0])
vectors = numpy.zeros((x_dim, y_dim), dtype=numpy.float32)
for i in range(0, len(encodings)):
    vectors[i] = numpy.asarray(encodings[i])
print("start indexing")
index = DenseFlatIndexer(vector_sz=y_dim)
# index=DenseHNSWFlatIndexer(vector_sz=y_dim)
index.index_data(vectors)
index.index_id_to_db_id = doc_indices
print(index.index.ntotal)
#faiss.write_index(index.index,"faiss-index-ms_marco")
found=index.search(query_encodings,10)

print("finished indexing")
print("index entities")
all_found = 0
all_not_found = 0
for i in range(0,len(relevants)):
    #correct_entities = [relevants[i]]
    prediction = found[i]
    if relevants[i] in prediction:
        all_found += 1
    else:
        all_not_found += 1
print("found:" + str(all_found) + " not found:" + str(all_not_found))
recall_at_10 = all_found / (all_not_found + all_found)
all_rr = []  # List to store reciprocal ranks for MRR calculation
print("Recall@10: " + str(recall_at_10))
for i in range(len(relevants)):
    #correct_entities = set(self.doc_to_ent[self.documents[i]])  # Set of correct entities
    prediction = found[i]  # Predicted entities

    # Find the rank of the first correct entity
    reciprocal_rank = 0
    for rank, entity in enumerate(prediction, start=1):
        if entity == relevants[i]:
            reciprocal_rank = 1 / rank
            break

    all_rr.append(reciprocal_rank)

mrr = sum(all_rr) / len(all_rr) if all_rr else 0
print(f"Mean Reciprocal Rank (MRR): {mrr:.5f}")





