from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import pandas as pd
import numpy as np
import random

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
        sent=row['sentence'][:-1] # remove period
        sents[sent_id][gender]['sentence']=sent
        if noun1=="someone" and ("someone" not in sent.split()):
                noun1="Someone"
        if noun2=="someone" and ("someone" not in sent.split()):
                noun2="Someone"
        sents[sent_id][gender]['noun1']=noun1
        sents[sent_id][gender]['noun2']=noun2
        for p in pronouns[gender]:
            if p in sent.split():
                sents[sent_id][gender]['pronoun']=sent.split().index(p)
                break
        if int(answer)==0:
            sents[sent_id][gender]['antecedent']=sent.split().index(noun1)
        else:
            sents[sent_id][gender]['antecedent']=sent.split().index(noun2)

    print("Dataset parsed succesfully!")
    return sents

def evaluate(predictor, sents):
    mismatches=0
    male_correct=female_correct=0
    for sent in sents:
        male_pred=female_pred=-1 # -1: don't know, 0: incorrect, 1: correct

        res_male=predictor.predict(document=sent["male"]["sentence"])
        pronoun_span_idx=-1
        for i in range(len(res_male["top_spans"])):
            if sent["male"]["pronoun"] in res_male["top_spans"][i]:
                pronoun_span_idx=i
                break
        antecedent_span_idx=-1
        for i in range(len(res_male["top_spans"])):
            if sent["male"]["antecedent"] in res_male["top_spans"][i]:
                antecedent_span_idx=i
                break
        if pronoun_span_idx!=-1 and antecedent_span_idx!=-1:
            male_pred=1 if (res_male["predicted_antecedents"][pronoun_span_idx]==antecedent_span_idx) else 0
            if male_pred==1: male_correct+=1

        res_female=predictor.predict(document=sent["female"]["sentence"])
        pronoun_span_idx=-1
        for i in range(len(res_female["top_spans"])):
            if sent["female"]["pronoun"] in res_female["top_spans"][i]:
                pronoun_span_idx=i
                break
        antecedent_span_idx=-1
        for i in range(len(res_female["top_spans"])):
            if sent["female"]["antecedent"] in res_female["top_spans"][i]:
                antecedent_span_idx=i
                break
        if pronoun_span_idx!=-1 and antecedent_span_idx!=-1:
            female_pred=1 if (res_female["predicted_antecedents"][pronoun_span_idx]==antecedent_span_idx) else 0
            if female_pred==1: female_correct+=1

        if male_pred!=female_pred: mismatches+=1
    print("Mismatch rate: ", mismatches/len(sents))
    print("Male accuracy: ", male_correct/len(sents))
    print("Female accuracy: ", female_correct/len(sents))
    return mismatches/len(sents)

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
print("Model loaded succesfully!")

# replace with appropriate alternate construction (e.g. synonymized.tsv)
sents = load_dataset('data/winogender.tsv')
test_set=[]
for sent_id in sents:
    test_set.append(sents[sent_id])
mismatch_rate=evaluate(predictor, test_set)
print(mismatch_rate)


