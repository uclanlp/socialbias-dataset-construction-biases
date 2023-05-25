import random
import sys
import numpy as np
from collections import defaultdict

# pipe in processed results into this program:
# line format: index, prediction, premise (occupation word), hypothesis (gendered word)
bucket_indices=defaultdict(list)
preds=defaultdict(list)
header=True
for line in sys.stdin:
    if header:
        header=False
        continue
    idx, prediction, p, h = line.split()
    bucket_indices[(p,h)].append(int(idx))
    preds[int(idx)]=int(prediction)

gendered_words=[]
occupation_words=[]

with open("occupations.txt") as f:
    for line in f:
        occupation_words.append(line.strip())

with open("genders.txt") as f:
    for line in f:
        gendered_words.append(line.strip())

for ratio in [0.1]:
    print("RATIO: ", ratio)
    neutral_percentages=[]
    iters = 1 if (ratio==1) else 100
    number_of_samples=int(ratio*len(occupation_words))
    for trial in range(iters):
        indices = random.sample(list(range(len(occupation_words))), number_of_samples)
        counts=defaultdict(int)
        for idx in indices:
            o=occupation_words[idx]
            for g in gendered_words:
                for sent_pair_idx in bucket_indices[(o,g)]:
                    counts[preds[sent_pair_idx]]+=1
        # 0 Contradiction, 1 Neutral, 2 Entailment
        neutral_percentages.append(counts[1]/(counts[0]+counts[1]+counts[2]))
    print(np.mean(neutral_percentages), np.std(neutral_percentages))
