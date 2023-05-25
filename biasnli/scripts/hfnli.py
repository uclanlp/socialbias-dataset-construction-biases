import torch
from sentence_transformers import CrossEncoder
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def split_list(l, parts=1):
    length = len(l)
    return [ l[i*length // parts: (i+1)*length // parts]
             for i in range(parts) ]

# change to the appropriate model (say alberta)
model = CrossEncoder('cross-encoder/nli-distilroberta-base')

label_mapping = ['contradiction', 'entailment', 'neutral']

# change to the appropriate alternate construction (say negation.csv)
path="../data/occupationgender.csv"
df = pd.read_csv(path, usecols=['premise','hypothesis'],header=0)

counts=defaultdict(int)
sents=[]

for idx , row in df.iterrows():
    sents.append((row['premise'],row['hypothesis']))

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

preds=[]
sent_batches=split_list(sents, len(sents)//64)
for batch in sent_batches:
    with torch.no_grad():
        scores = model.predict(batch)
        labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
        preds=preds+labels
        for el in labels:
            counts[el]+=1
print(counts)
for i in range(len(preds)):
    print(i,preds[i])