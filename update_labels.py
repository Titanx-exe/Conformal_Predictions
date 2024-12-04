import pickle
dc=pickle.load(open("data/entity_descriptions.pkl","rb"))
all_labels={}
for k in dc.keys():
    lb=dc[k].replace("label:","title")
    lb=lb.replace("alt","")
    while "[SEP]"in lb:
        lb = lb.replace("[SEP]", "")
    all_labels[k]=lb
pickle.dump(all_labels,open("data/ent_descriptions_update.pkl","wb"))