corpus_sentences = [
    "I need to go to the bank to deposit some money.",
    "She opened a new savings account at the bank yesterday.",
    "The bank approved his loan application very quickly.",
    "There was a long queue at the bank during lunchtime.",
    "He withdrew cash from the bank before going on vacation.",
    "They invested their inheritance in a large international bank.",
    "We sat on the bank of the river and watched the boats go by.",
    "The children were playing on the grassy bank of the stream.",
    "Floods damaged the houses built near the river bank.",
    "They walked along the bank of the canal at sunset.",
    "The fisherman stood silently on the bank, waiting for a bite.",
    "He left his car parked close to the bank of the lake.",
    "The central bank decided to increase the interest rates.",
    "Online banking has reduced the need to visit a physical bank.",
    "The company’s headquarters are located near the river bank.",
    "She slipped on the muddy bank and almost fell into the water.",
    "The bank offered him a better interest rate on his mortgage.",
    "They organized a picnic on the river bank under the old tree.",
    "During the storm, the water rose and overflowed the bank.",
    "The robbery at the bank was reported on the evening news."
]

# STEP 1: LESK BASED SEEDING ON CORPUS

#i. convert the raw text into wsd instances
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from collections import Counter
import numpy as np
sent = "I went to the bank to deposit money."
tokens = word_tokenize(sent)
target_word='bank'

def extract_instances(corpus_sentences, target):
    instances=[]
    for doc_id, sent in enumerate(corpus_sentences):
        tokens=word_tokenize(sent)
        for i, tok in enumerate(tokens):
            if tok.lower()==target.lower():
               instances.append({
                    "doc_id": doc_id,
                    "sentence": sent,
                    "tokens": tokens,
                    "target_index": i
                })
    return instances

instances=extract_instances(corpus_sentences,target_word)

'''
ii.
For each occurrence of the target word, run Lesk for all senses, compute how well each sense’s 
definition matches the context, then take the difference between the best and second-best matches. If that difference (margin) is large, we treat this occurrence as a very confident Lesk decision and use it as a 
seed example for bootstrapping.
'''

def lesk_overlap_score(tokens,target,pos='n'):
    synsets=wn.synsets(target,pos=pos)
    context=set(w.lower() for w in tokens)
    scores=[]
    for syn in synsets:
        gloss_tokens=set(word_tokenize(syn.definition().lower()))
        score=len(context & gloss_tokens)
        scores.append((syn,score))
    return scores  # list of (synset, score)

def lesk_contrastive_seed_score(tokens,target,pos='n'):
    scores=lesk_overlap_score(tokens,target,pos)
    if not scores:
        return None,0,0
    scores_sorted=sorted(scores, key=lambda x:x[1], reverse=True)
    best_syn,best=scores_sorted[0]
    second=scores_sorted[1][1] if len(scores_sorted) > 1 else 0
    margin=best-second
    return best_syn, best,margin

#iii Select the high confidence seeds

def generate_seeds(instances, target, pos='n',margin_percentile=90):
    margins=[]
    temp=[]
    for inst in instances:
        syn,score,margin=lesk_contrastive_seed_score(inst["tokens"], target, pos)
        temp.append((inst,syn,score,margin))
        margins.append(margin)
    thr=np.percentile(margins, margin_percentile)
    
    seeds=[]
    unlabeled=[]
    for inst,syn,score, margin in temp:
        if syn is not None and margin>=thr:
            inst_copy=inst.copy()
            inst_copy["sense"]=syn
            inst_copy["margin"]=margin
            seeds.append(inst_copy)
        else:
            unlabeled.append(inst)
    return seeds, unlabeled

seeds, unlabeled=generate_seeds(instances,target_word, pos='n',margin_percentile=90)
print("Seeds:",len(seeds),"Unlabeled:",len(unlabeled))