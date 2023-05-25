# for use after downloading model from https://github.com/shtoshni/fast-coref
import sys
sys.path.append('fast-coref/src')
from inference.model_inference import Inference
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

inference_model = Inference("./", encoder_name="shtoshni/longformer_coreference_ontonotes")
print("Model loaded succesfully!")

pronouns={}
pronouns["female"] = ["she","her"]
pronouns["male"]= ["he","his","him"]
pronouns["neutral"]=["they","their","them"]


def load_dataset(path):
    df = pd.read_csv(path, sep='\t', header=0)
    sents={}

    for idx , row in df.iterrows():
        noun1,noun2,answer,gender,_=row['sentid'].split('.')
        sent_id=idx//3 # dataset consists of 3 sentences (male, female, neutral) for each example
        if sent_id not in sents: sents[sent_id]={}
        if gender not in sents[sent_id]: sents[sent_id][gender]={}
        sent=row['sentence']
        sents[sent_id][gender]['sentence']=sent
        if noun1=="someone" and ("someone" not in sent.split()):
                noun1="Someone"
        if noun2=="someone" and ("someone" not in sent.split()):
                noun2="Someone"
        sents[sent_id][gender]['noun1']=noun1
        sents[sent_id][gender]['noun2']=noun2
        for p in pronouns[gender]:
            if p in sent.split():
                sents[sent_id][gender]['pronoun']=p
                break
        if int(answer)==0:
            sents[sent_id][gender]['antecedent']=noun1
        else:
            sents[sent_id][gender]['antecedent']=noun2

    print("Dataset parsed succesfully!")
    return sents

def same_cluster(clusters, pronoun, noun):
  pronoun_cluster=None
  for cluster in clusters:
    for phrase in cluster:
      if phrase[1]==pronoun:
        pronoun_cluster=cluster
        break
    if pronoun_cluster is not None:
      break

  if pronoun_cluster is None:
    return False

  for phrase in pronoun_cluster:
    if noun in phrase[1]:
      return True

  return False

def evaluate(sents):
    mismatches=0
    male_correct=female_correct=0
    for sent in sents:
        male_pred=female_pred=-1 # -1: don't know, 0: incorrect, 1: correct

        res_male=inference_model.perform_coreference(sent["male"]["sentence"])
        male_pred=1 if same_cluster(res_male['clusters'],sent["male"]["pronoun"],sent["male"]["antecedent"]) else 0
        if male_pred==1: male_correct+=1

        res_female=inference_model.perform_coreference(sent["female"]["sentence"])
        female_pred=1 if same_cluster(res_female['clusters'],sent["female"]["pronoun"],sent["female"]["antecedent"]) else 0
        if female_pred==1: female_correct+=1

        if male_pred!=female_pred: mismatches+=1
    print("Mismatch rate: ", mismatches/len(sents))
    print("Male accuracy: ", male_correct/len(sents))
    print("Female accuracy: ", female_correct/len(sents))
    return mismatches/len(sents)

# replace with appropriate alternate construction (e.g. synonymized.tsv)
sents = load_dataset('data/winogender.tsv')
test_set=[]
for sent_id in sents:
    test_set.append(sents[sent_id])
mismatch_rate=evaluate(test_set)
print(mismatch_rate)
