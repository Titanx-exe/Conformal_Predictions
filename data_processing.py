from tqdm import tqdm
import json
import random
from SPARQLWrapper import SPARQLWrapper, JSON
import pickle
from nif import NIFDocument,NIFContent



def process_lcquad_file(file_path):
    input=json.load(open(file_path,"r",encoding="utf-8"))
    entities={}
    documents=set()
    doc_to_ent={}
    for sample in input:
        if "entities" in sample and sample["question"] is not None:
            doc_to_ent[sample["question"]] = set()
            for ent in sample["entities"]:
                if not ent["uri"] in entities:
                    entities[ent["uri"]]=set()
                doc_to_ent[sample["question"]].add(ent["uri"])
                entities[ent["uri"]].add(sample["question"])
            documents.add(sample["question"])
    return entities,documents,doc_to_ent

def process_minitaka_file(file_path):
    input=json.load(open(file_path,"r",encoding="utf-8"))
    entities={}
    documents=set()
    doc_to_ent={}
    for sample in input:
        if "questionEntity" in sample and sample["question"] is not None:
            question_entities=sample["questionEntity"]
            for ent in question_entities:
                if ent["entityType"]=="entity"and ent["name"] is not None:
                    doc_to_ent[sample["question"]] = set()
                    if not "http://www.wikidata.org/entity/"+ent["name"] in entities:
                        entities["http://www.wikidata.org/entity/"+ent["name"]]=set()
                    doc_to_ent[sample["question"]].add("http://www.wikidata.org/entity/"+ent["name"])
                    entities["http://www.wikidata.org/entity/"+ent["name"]].add(sample["question"])
                    documents.add(sample["question"])
    return entities,documents,doc_to_ent

def create_batch_ent(entity,pos_docs,documents:list,doc_to_ent,batch_size=10):
    batch=[]
    pos_document=random.choice(pos_docs)
    batch.append((entity,pos_document))
    covered_entities=set()
    covered_entities = set(doc_to_ent[pos_document])
    for i in range(1,batch_size):
        for doc in documents:
            doc_ents=doc_to_ent[doc]
            if len(doc_ents.intersection(covered_entities))==0:
                if len(doc_ents)>0:
                    b_ent=random.choice(list(doc_ents))
                    batch.append((b_ent,doc))
                    covered_entities=covered_entities.union(doc_ents)
                    break
    return batch

def create_batch_index(entity,entities,pos_docs,document_encodings:dict,entity_index,doc_to_ent,batch_size=10):
    batch=[]
    pos_document=random.choice(pos_docs)
    batch.append((entity,pos_document))
    #covered_entities=set()
    covered_entities = set(doc_to_ent[pos_document])
    candidate_entities=entity_index.search([document_encodings[pos_document]],100)[0]
    documents=[]
    for ent in candidate_entities:
        if ent in entities:
            documents.extend(list(entities[ent]))
    for i in range(1,batch_size):
        for doc in documents:
            doc_ents=doc_to_ent[doc]
            if len(doc_ents.intersection(covered_entities))==0:
                if len(doc_ents)>0:
                    b_ent=random.choice(list(doc_ents))
                    batch.append((b_ent,doc))
                    covered_entities=covered_entities.union(doc_ents)
                    break
    return batch

def create_batch_index_document(document,entities,document_encodings:dict,entity_index,doc_to_ent,batch_size=10):
    batch=[]
    pos_entities=list(doc_to_ent[document])
    pos_ent=random.choice(pos_entities)
    batch.append((pos_ent,document))
    #covered_entities=set()
    covered_entities = set(pos_entities)
    candidate_entities=entity_index.search([document_encodings[document]],100)[0]
    documents=[]
    for ent in candidate_entities:
        if ent in entities:
            documents.extend(list(entities[ent]))
    for i in range(1,batch_size):
        for doc in documents:
            doc_ents=doc_to_ent[doc]
            if len(doc_ents.intersection(covered_entities))==0:
                if len(doc_ents)>0:
                    b_ent=random.choice(list(doc_ents))
                    batch.append((b_ent,doc))
                    covered_entities=covered_entities.union(doc_ents)
                    break
    return batch

def get_text_knowledge_for_entities(ent_list):


    sparql = SPARQLWrapper(
        "https://20230607-truthy.wikidata.data.dice-research.org/sparql/"

    )
    sparql.setReturnFormat(JSON)
    ent_list=["wd:"+el.replace("http://www.wikidata.org/entity/","") for el in ent_list]
    ent_str=" ".join(ent_list)
    # gets the first 3 geological ages
    # from a Geological Timescale database,
    # via a SPARQL endpoint
    query="""
        PREFIX wd: <http://www.wikidata.org/entity/>
        select distinct ?resource ?label ?alt ?desc where {
            {?resource <http://www.w3.org/2000/01/rdf-schema#label> ?label.
            FILTER (lang(?label) = "en")}
            UNION
            {?resource 	<http://www.w3.org/2004/02/skos/core#altLabel>?alt.
            FILTER (lang(?alt) = "en")
            }UNION
            {?resource 	<http://schema.org/description>?desc.
            FILTER (lang(?desc) = "en")
            }
            VALUES ?resource {"""+ent_str+"""}
        }

                    """
    #print(query)
    sparql.setQuery(query)

    ret = sparql.queryAndConvert()
    found_res={}
    for r in ret["results"]["bindings"]:
        uri=r["resource"]["value"]
        if not uri in found_res:
            found_res[uri]={"label":[],"alt":[],"desc":[]}
        if "alt"in r:
            found_res[uri]["alt"].append(r["alt"]["value"])
        if "label"in r:
            found_res[uri]["label"].append(r["label"]["value"])
        if "desc"in r:
            found_res[uri]["desc"].append(r["desc"]["value"])
    uri_text_dict={}
    for key in found_res.keys():
        text_desc=""
        if len(found_res[key]["label"])>0:
            text_desc+="label: "+", ".join(found_res[key]["label"])+"[SEP]"
        if len(found_res[key]["alt"])>0:
            text_desc+="alt: "+", ".join(found_res[key]["alt"])+"[SEP]"
        if len(found_res[key]["desc"])>0:
            text_desc+="desc: "+", ".join(found_res[key]["desc"])+"[SEP]"
        uri_text_dict[key]=text_desc
    return uri_text_dict


class Aida_joint_el():

    def __call__(self, features):
        return self.process_sample([el["input"]for el in features],[el["label"]for el in features])
    def load(self,path_to_ds):
        with open(path_to_ds, 'r', encoding='utf-8') as file:
            doc = NIFDocument.nifStringToNifDocument(file.read())
        return doc


    def groupNifDocumentByRefContext(self,ds: NIFDocument):
        refContextToDocument = {}
        for nifContent in ds.nifContent:
            if nifContent.reference_context is not None:
                docrefcontext = nifContent.reference_context
            else:
                docrefcontext = nifContent.uri
            if docrefcontext in refContextToDocument:
                refContextToDocument.get(docrefcontext).addContent(nifContent)
            else:
                doc = NIFDocument.NIFDocument()
                doc.addContent(nifContent)
                refContextToDocument[nifContent.uri] = doc
        #for el in refContextToDocument:
        #    refContextToDocument[el].nifContent.sort(key=lambda x: x.begin_index)
        return refContextToDocument

    def split(self,a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    def read_ds_to_list(self,path_to_ds,tokenizer):
        # candidate_file=json.load(open("../candiatedata/"+dsName+"_candidates.json"))
        #wikipedia_data = pickle.load(open("../data/wikipedia_data.pkl", "rb"))
        # labels=pickle.load(open("labels_full_updated.sav","rb"))
        documentmap = self.groupNifDocumentByRefContext(self.load(path_to_ds))
        samples = []
        seq_len = 50
        i = 0
        documents=set()
        doc_to_ent={}
        entities={}
        for el in documentmap:
            source = ""
            target = ""
            annotations = []
            for cont in documentmap.get(el).nifContent:
                if cont.is_string is not None:
                    target = cont.is_string
                    source = cont.is_string
                else:
                    annotations.append(cont)
            tokens = tokenizer.tokenize(source)
            tk_ids=tokenizer.convert_tokens_to_ids(tokens)
            curr_seq_len=len(tk_ids)
            num_splits = curr_seq_len // seq_len

            if num_splits > 0:
                sentences = source.split(". ")
                splits = []
                chunks = list(self.split(sentences, num_splits + 1))
                for chunk in chunks:
                    splits.append(". ".join(chunk) + ". ")
            else:
                splits = [source]
            annotations.sort(key=lambda x: int(x.begin_index))
            target_splits_el = splits.copy()
            # target_splits_ner = splits.copy()
            offsets = [0 for el in splits]
            starttag = "[START_ENT] "
            endtag = " [END_ENT]"
            # cands = set()
            elinks = set()

            for an in annotations:
                if not "notInWiki" in an.taIdentRef:
                    for sp in splits:
                        if not int(an.begin_index) > len(sp):
                            if not sp in doc_to_ent:
                                doc_to_ent[sp]=set()
                            if not an.taIdentRef in entities:
                                entities[an.taIdentRef]=set()
                            entities[an.taIdentRef].add(sp)
                            doc_to_ent[sp].add(an.taIdentRef)
                            documents.add(sp)
                        # sp_id = splits.index(sp)


        return entities, documents, doc_to_ent
'''
entities,documents,doc_to_ent=process_lcquad_file("data/train/lcquad.json")
entities_test,_,_=process_lcquad_file("data/test/lcquad.json")
entities.update(entities_test)



ent_label_dict={}
batch=[]
for ent in tqdm(entities.keys()):

    if len(batch)<50:
        batch.append(ent)
    else:
        ent_dict=get_text_knowledge_for_entities(batch)

        ent_label_dict.update(ent_dict)
        batch=[]
if len(batch)>0:
    ent_dict=get_text_knowledge_for_entities(batch)
    ent_label_dict.update(ent_dict)
pickle.dump(ent_label_dict, open("data/entity_descriptions.pkl", "wb"))
'''