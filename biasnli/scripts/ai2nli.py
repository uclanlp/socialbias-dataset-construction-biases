from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

pred_to_idx={"entailment":0, "neutral":1, "contradiction":2}

def split_list(l, parts=1):
    length = len(l)
    return [ l[i*length // parts: (i+1)*length // parts]
             for i in range(parts) ]

#ELMo-based Decomposable Attention
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz")

# change to the appropriate alternate construction (say negation.csv)
path="../data/occupationgender.csv"
df = pd.read_csv(path, usecols=['premise','hypothesis'],header=0)

counts=defaultdict(int)
sents=[]

for idx , row in df.iterrows():
    sents.append({"premise":row['premise'], "hypothesis":row['hypothesis']})

i=0
sent_batches=split_list(sents, len(sents)//64)
for batch in sent_batches:
    res=predictor.predict_batch_json(batch)
    for el in res:
        counts[el['label']]+=1
        print(i,pred_to_idx[el['label']])
        i+=1

print(counts)

def pred(data):
    final_res=[]
    print(data)
    print(len(data))
    for sent_pair in data:
        p,h=sent_pair.split("#")
        res=predictor.predict(premise=p,hypothesis=h)
        final_res.append(res["label_probs"])
    return final_res
