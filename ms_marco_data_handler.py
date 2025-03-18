import gzip
import pickle
import glob
import torch
from tqdm import tqdm
from parameters import RankingParser
from torch.utils.data import DataLoader
import random
from indexing import index_entities, index_data, index_queries
#from msmarco_preprocessing import queries
from models.BiEncoderHuggingface import BiEncoderRanker
from MS_marco_collator import Biencoder_Collator
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

class Ms_marco_data_handler:
    def __init__(self,params):
        self.queries=load_queries("data/msmarco/msmarco-doctrain-queries.tsv.gz","data/msmarco/msmarco-doctrain-qrels.tsv.gz")
        self.gold_documents=pickle.load(open("data/msmarco/gold_documents","rb"))
        self.params=params
        self.data_splits=glob.glob("data/msmarco/doc_dictionary_split*")
        self.current_split=None
        self.current_doc_encodings=None
        #self.current_doc_index=None
        self.current_query_index=None
        self.current_documents=[]
        self.current_queries=None
        self.curr_negative_docs=[]
        print("loaded")

    def encode_documents(self,documents, model, collator):
        #documents = list(documents)
        data_loader = DataLoader(documents, shuffle=True, batch_size=25,
                                 collate_fn=collator.collate_entities)
        iter_ = tqdm(data_loader, desc="Encode Train Documents")
        doc_encodings = []
        for step, batch in enumerate(iter_):
            encodings = model.encode_candidate(batch).tolist()
            doc_encodings.extend(encodings)
        encoding_map = {}
        for i in range(len(doc_encodings)):
            encoding_map[documents[i]] = doc_encodings[i]
        return encoding_map

    def reload_current_data(self,model,collator,reload_full=False):
        model.eval()
        if reload_full:
            current_split=pickle.load(open(random.choice(self.data_splits),"rb"))
            query_ids=random.sample(list(self.queries.keys()), self.params["max_queries_per_step"])
            doc_ids = random.sample(list(current_split.keys()), self.params["max_negatives_per_step"])
            self.curr_negative_docs=doc_ids
            all_docs=[]
            all_docs.extend(doc_ids)
            all_docs.extend([self.queries[el]["relevant"][0]for el in query_ids])
            current_split.update(self.gold_documents)
            self.current_queries=query_ids
            self.current_split={el:"title: "+current_split[el][1]+ "[SEP] content: "+current_split[el][2]for el in all_docs}

            collator.documents=self.current_split
        self.current_doc_encodings=self.encode_documents(self.curr_negative_docs,model,collator)
        self.current_query_index = index_queries(model,self.current_queries,  collator)

    def create_batch_index(self,num_queries=5,rand_docs=2,num_noise_labels=0):
        random_docs=random.sample(self.curr_negative_docs,rand_docs)
        #candidate_queries = self.current_query_index.search([self.current_doc_encodings[document]], num_queries)[0]
        candidate_queries = self.current_query_index.search([self.current_doc_encodings[rd] for rd in random_docs], num_queries)
        all_queries=[]

        for queries in candidate_queries:
            all_queries.extend(queries)
        all_queries=list(set(all_queries))
        correct_pairs={ind: self.queries[ind]["relevant"][0]for ind in all_queries}
        documents=[self.queries[ind]["relevant"][0]for ind in all_queries]
        documents.extend(random_docs)
        documents=list(set(documents))
        random.shuffle(documents)
        #pairs.extend([(ind,document,False)for ind in candidate_queries])
        #random.shuffle(pairs)
        labels=[documents.index(correct_pairs[query])for query in list(correct_pairs.keys())]
        for i in range(num_noise_labels):
            update_ind = labels[i]
            while update_ind == labels[i]:
                update_ind = random.randrange(len(documents))
            labels[i] = update_ind
        return all_queries,documents,labels
'''
device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
parser = RankingParser(add_model_args=True)
parser.add_training_args()
parser.add_eval_args()
args = parser.parse_args()
print(args)
params = args.__dict__
model=BiEncoderRanker(params)
model.to(device)
    #for biencoder
handler=Ms_marco_data_handler(params)
collator = Biencoder_Collator(tokenizer=model.tokenizer,args=params,queries=handler.queries, device=device)
handler.reload_current_data(model,collator)
for el in list(handler.curr_negative_docs):
    handler.create_batch_index(el)
print("fn")
'''
