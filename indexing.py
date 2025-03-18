from tqdm import tqdm
import numpy
import faiss
import torch
#from collator import Biencoder_Collator
from torch.utils.data import DataLoader
#import data_processing
from parameters import RankingParser
#from models.BiEncoder import BiEncoderRanker

class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = {}
        self.index = None

    def index_data(self, data: numpy.array):
        raise NotImplementedError

    def search_knn(self, query_vectors: numpy.array, top_docs: int):
        raise NotImplementedError

    def serialize(self, index_file: str):
        faiss.write_index(self.index, index_file)

    def deserialize_from(self, index_file: str):
        self.index = faiss.read_index(index_file)

    def search(self, candidate_encodings,k):
        candidates=numpy.array(candidate_encodings,dtype=numpy.float32)
        found_uris=[]
        D, I = self.search_knn(candidates, k)
        found=[]
        for el in numpy.nditer(I, order='C'):
            f = self.index_id_to_db_id[int(el)]
            found.append(f)
            if len(found) == k:
                found_uris.append(found)
                found = []
        return found_uris

# DenseFlatIndexer does exact search
class DenseFlatIndexer(DenseIndexer):
    def __init__(self, vector_sz: int = 1, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: numpy.array):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            vectors = [numpy.reshape(t, (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = numpy.concatenate(vectors, axis=0)
            self.index.add(vectors)


    def search_knn(self, query_vectors, top_k):
        scores, indexes = self.index.search(query_vectors, top_k)
        return scores, indexes


# DenseHNSWFlatIndexer does approximate search
class DenseHNSWFlatIndexer(DenseIndexer):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
        self,
        vector_sz: int,
        buffer_size: int = 50000,
        store_n: int = 128,
        ef_search: int = 256,
        ef_construction: int = 200,
    ):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)

        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, store_n)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index
        self.phi = 0

    def index_data(self, data: numpy.array):
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError(
                "DPR HNSWF index needs to index all data at once,"
                "results will be unpredictable otherwise."
            )
        phi = 0
        for i, item in enumerate(data):
            doc_vector = item
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        self.phi = 0

        # indexing in batches is beneficial for many faiss index types
        cnt = 0
        for i in range(0, n, self.buffer_size):
            vectors = [numpy.reshape(t, (1, -1)) for t in data[i : i + self.buffer_size]]

            norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
            aux_dims = [numpy.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [
                numpy.hstack((doc_vector, aux_dims[i].reshape(-1, 1)))
                for i, doc_vector in enumerate(vectors)
            ]
            hnsw_vectors = numpy.concatenate(hnsw_vectors, axis=0)

            self.index.add(hnsw_vectors)
            cnt += self.buffer_size


    def search_knn(self, query_vectors, top_k):
        aux_dim = numpy.zeros(len(query_vectors), dtype="float32")
        query_nhsw_vectors = numpy.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        scores, indexes = self.index.search(query_nhsw_vectors, top_k)
        return scores, indexes

    def deserialize_from(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize_from(file)
        # to trigger warning on subsequent indexing
        self.phi = 1

def generateVectors(entities,model,batchsize,collator):
    encodings=[]

    dataloader = DataLoader(
    entities, shuffle=False, batch_size=batchsize,collate_fn=collator.collate_entities
    )
    iter_ = tqdm(dataloader)
    for step, batch in enumerate(iter_):
        encs = model.encode_candidate(batch).tolist()
        encodings.extend(encs)
    return encodings

def generateVectors_generic(entities,batchsize,collate_fn,model_fn):
    encodings=[]

    dataloader = DataLoader(
    entities, shuffle=False, batch_size=batchsize,collate_fn=collate_fn
    )
    iter_ = tqdm(dataloader)
    for step, batch in enumerate(iter_):
        encs = model_fn(batch).tolist()
        encodings.extend(encs)
    return encodings

def generate_vectors_query(queries,model,batchsize,collator):
    encodings=[]

    dataloader = DataLoader(
    queries, shuffle=False, batch_size=batchsize,collate_fn=collator.collate_context
    )
    iter_ = tqdm(dataloader)
    for step, batch in enumerate(iter_):
        encs = model.encode_context(batch).tolist()
        encodings.extend(encs)
    return encodings


max_Id=0
print("load entities")


#def get_entitities():
#    entityLables = pickle.load(open("cleand_labels.sav", "rb"))
#    return entityLables

def index_queries(model,queries,collator):
    print("encode entities")
    model.eval()
    encodings = generate_vectors_query(queries,model,100,collator)
    idToIndex={}
    print("start geneationg vectors")
    x_dim=len(encodings)
    y_dim=len(encodings[0])
    vectors=numpy.zeros((x_dim,y_dim),dtype=numpy.float32)
    for i in range(0,len(encodings)):
        vectors[i] = numpy.asarray(encodings[i])
        idToIndex.update({i:queries[i]})
    print("start indexing")
    index=DenseHNSWFlatIndexer(vector_sz=y_dim)
    index.index_data(vectors)
    index.index_id_to_db_id=idToIndex
    print(index.index.ntotal)
    faiss.write_index(index.index,"faiss-hswf-index-test")
    print("finished")
    print("index entities")
    return index

def index_entities(model,entities,collator):
    print("encode entities")
    model.eval()
    encodings = generateVectors(entities,model,25,collator)
    idToIndex={}
    print("start geneationg vectors")
    x_dim=len(encodings)
    y_dim=len(encodings[0])
    vectors=numpy.zeros((x_dim,y_dim),dtype=numpy.float32)
    for i in range(0,len(encodings)):
        vectors[i] = numpy.asarray(encodings[i])
        idToIndex.update({i:entities[i]})
    print("start indexing")
    DenseFlatIndexer
    index = DenseFlatIndexer(vector_sz=y_dim)
    #index=DenseHNSWFlatIndexer(vector_sz=y_dim)
    index.index_data(vectors)
    index.index_id_to_db_id=idToIndex
    print(index.index.ntotal)
    faiss.write_index(index.index,"faiss-hswf-index-test")
    print("finished")
    print("index entities")
    return index

def index_data(model,entities,collate_fn,model_fn):
    print("encode entities")
    model.eval()
    encodings = generateVectors_generic(entities,200,collate_fn,model_fn)
    idToIndex={}
    print("start geneationg vectors")
    x_dim=len(encodings)
    y_dim=len(encodings[0])
    vectors=numpy.zeros((x_dim,y_dim),dtype=numpy.float32)
    for i in range(0,len(encodings)):
        vectors[i] = numpy.asarray(encodings[i])
        idToIndex.update({i:entities[i]})
    print("start indexing")
    DenseFlatIndexer
    index = DenseFlatIndexer(vector_sz=y_dim)
    #index=DenseHNSWFlatIndexer(vector_sz=y_dim)
    index.index_data(vectors)
    index.index_id_to_db_id=idToIndex
    print(index.index.ntotal)
    #faiss.write_index(index.index,"faiss-hswf-index-test")
    print("finished")
    print("index entities")
    return index,idToIndex

device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
parser = RankingParser(add_model_args=True)
parser.add_training_args()
parser.add_eval_args()
'''
# args = argparse.Namespace(**params)
args = parser.parse_args()
print(args)
params = args.__dict__
model=BiEncoderRanker(params)
model.to(device)
collator = Biencoder_Collator(tokenizer=model.tokenizer,args=params, device=device)
entities,_,doc_to_ent=data_processing.process_lcquad_file("data/test/lcquad.json")
entities_train,_,_=data_processing.process_lcquad_file("data/train/lcquad.json")
entities=list(entities.keys())
entities.extend(list(entities_train.keys()))
index=index_entities(model,entities,collator)
#seq,_,_=collator.process_sample("Test Sequence")
#enc=torch.tensor(model.encode_context(torch.tensor([seq],device=device)))
#entities=index.search(enc,10)
for doc in doc_to_ent:
    seq, _, _ = collator.process_sample(doc)
    enc = torch.tensor(model.encode_context(torch.tensor([seq], device=device)))
    corr=doc_to_ent[doc]
    entities = index.search(enc, 10)
    print(entities)
'''
