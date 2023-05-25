from transformers import T5Tokenizer, T5ForConditionalGeneration
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
        question=sents[sent_id][gender]['sentence']
        question+=" Who does the word '{0}' refer to? \\n ".format(sents[sent_id][gender]['pronoun'])
        question+="(a) {0} (b) {1}".format(sents[sent_id][gender]['noun1'], sents[sent_id][gender]['noun2'])
        sents[sent_id][gender]['question']=question

    print("Dataset parsed succesfully!")
    return sents

model_name = "allenai/unifiedqa-t5-large" # specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
print("Large Model loaded succesfully!")


def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)

def evaluate(sents):
    mismatches=0
    male_correct=female_correct=0
    for sent in sents:
        male_pred=female_pred=-1 # -1: don't know, 0: incorrect, 1: correct

        res_male=run_model(sent["male"]["question"])
        male_pred=1 if (res_male[0]==sent["male"]['antecedent']) else 0
        if male_pred==1: male_correct+=1

        res_female=run_model(sent["female"]["question"])
        female_pred=1 if (res_female[0]==sent["female"]['antecedent']) else 0
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

