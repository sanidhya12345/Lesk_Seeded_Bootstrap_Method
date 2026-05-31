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

# Stage-2 yarowsky style bootstrapping with naive bayes classifier

#i.finding out the context features

from sklearn.feature_extraction.text import CountVectorizer

WINDOW = 5  # left/right window size

def context_string(inst):
    tokens = inst["tokens"]
    i = inst["target_index"]
    left = tokens[max(0, i-WINDOW):i]
    right = tokens[i+1:i+1+WINDOW]
    ctx = left + right
    return " ".join(w.lower() for w in ctx)

#ii. training of naive bayes using initial seeds
from sklearn.naive_bayes import MultinomialNB

# 1) Seed texts + labels
seed_texts = [context_string(inst) for inst in seeds]

seed_labels = [
    inst["sense"].name() if hasattr(inst["sense"], "name") else inst["sense"]
    for inst in seeds
]

# 2) Vectorizer fit on seed contexts
vectorizer = CountVectorizer()
X_seed = vectorizer.fit_transform(seed_texts)

# 3) Train Naive Bayes
clf = MultinomialNB()
clf.fit(X_seed, seed_labels)

import numpy as np

def classify_unlabeled(unlabeled, vectorizer, clf, conf_thr=0.9):
    new_labeled = []
    still_unlabeled = []
    for inst in unlabeled:
        ctx = context_string(inst)
        x = vectorizer.transform([ctx])
        probs = clf.predict_proba(x)[0]
        max_idx = np.argmax(probs)
        max_prob = probs[max_idx]
        if max_prob >= conf_thr:
            inst_copy = inst.copy()
            inst_copy["sense"] = clf.classes_[max_idx]
            inst_copy["conf"] = float(max_prob)
            new_labeled.append(inst_copy)
        else:
            still_unlabeled.append(inst)
    return new_labeled, still_unlabeled

new_labeled, remaining = classify_unlabeled(unlabeled, vectorizer, clf, conf_thr=0.9)
print("New labeled from NB:", len(new_labeled), "Remaining unlabeled:", len(remaining))

def yarowsky_bootstrap(seeds, unlabeled, max_iter=5, conf_thr=0.9, min_new=1):
    labeled = seeds[:]       # start from Lesk seeds
    unl = unlabeled[:]       # remaining instances

    # initial train
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([context_string(inst) for inst in labeled])
    y = [
        inst["sense"].name() if hasattr(inst["sense"], "name") else inst["sense"]
        for inst in labeled
    ]

    for it in range(max_iter):
        clf = MultinomialNB()
        clf.fit(X, y)

        new_labeled, unl = classify_unlabeled(unl, vectorizer, clf, conf_thr=conf_thr)
        print(f"Iter {it}: new_labeled = {len(new_labeled)}, remaining = {len(unl)}")

        if len(new_labeled) < min_new:
            print("Stopping: too few new high-confidence labels.")
            break

        # add new labeled examples to training set
        labeled.extend(new_labeled)
        X = vectorizer.fit_transform([context_string(inst) for inst in labeled])
        y = [
            inst["sense"].name() if hasattr(inst["sense"], "name") else inst["sense"]
            for inst in labeled
        ]

    return labeled, unl, vectorizer, clf

final_labeled, final_unlabeled, vectorizer, clf = yarowsky_bootstrap(
    seeds, unlabeled, max_iter=5, conf_thr=0.7, min_new=1
)

def pretty_print_results(labeled, corpus_sentences):
    for inst in labeled:
        doc_id = inst["doc_id"]
        sent = inst["sentence"]
        sense_val = inst["sense"]

        # sense_val Synset hai ya string?
        if hasattr(sense_val, "definition"):      # Synset object
            syn = sense_val
            sense_name = syn.name()
            gloss = syn.definition()
        else:                                     # string label, e.g. 'bank.n.09'
            sense_name = sense_val
            try:
                syn = wn.synset(sense_name)
                gloss = syn.definition()
            except:
                syn = None
                gloss = "N/A"

        print(f"[doc {doc_id}] {sent}")
        print(f"  -> sense: {sense_name}")
        print(f"     gloss: {gloss}\n")
print("=== Final labeled instances ===")
pretty_print_results(final_labeled, corpus_sentences)
print("Remaining unlabeled:", len(final_unlabeled))