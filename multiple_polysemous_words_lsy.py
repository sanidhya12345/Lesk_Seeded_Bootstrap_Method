import nltk
from nltk.corpus import wordnet as wn, semcor
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Make sure required NLTK data is available (run once)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('semcor')


# -----------------------------
# 1. Helper: get SemCor sentences for a target word
# -----------------------------
def get_semcor_sentences_for_word(word, max_sent=10):
    """
    Return up to max_sent raw sentences from SemCor that contain the target word.
    """
    sentences = []
    for sent in semcor.sents():
        tokens_lower = [t.lower() for t in sent]
        if word.lower() in tokens_lower:
            sentences.append(" ".join(sent))
            if len(sentences) >= max_sent:
                break
    return sentences


# -----------------------------
# 2. Build WSD instances for a word
# -----------------------------
def extract_instances(corpus_sentences, target):
    instances = []
    for doc_id, sent in enumerate(corpus_sentences):
        tokens = word_tokenize(sent)
        for i, tok in enumerate(tokens):
            if tok.lower() == target.lower():
                instances.append(
                    {
                        "doc_id": doc_id,
                        "sentence": sent,
                        "tokens": tokens,
                        "target_index": i,
                    }
                )
    return instances


# -----------------------------
# 3. Lesk-based contrastive seeding
# -----------------------------
def lesk_overlap_scores(tokens, target, pos="n"):
    synsets = wn.synsets(target, pos=pos)
    context = set(w.lower() for w in tokens)
    scores = []
    for syn in synsets:
        gloss_tokens = set(word_tokenize(syn.definition().lower()))
        score = len(context & gloss_tokens)
        scores.append((syn, score))
    return scores  # list of (synset, score)


def lesk_contrastive_seed_score(tokens, target, pos="n"):
    scores = lesk_overlap_scores(tokens, target, pos)
    if not scores:
        return None, 0, 0
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    best_syn, best = scores_sorted[0]
    second = scores_sorted[1][1] if len(scores_sorted) > 1 else 0
    margin = best - second
    return best_syn, best, margin


def generate_seeds(instances, target, pos="n", margin_percentile=90):
    margins = []
    temp = []
    for inst in instances:
        syn, score, margin = lesk_contrastive_seed_score(inst["tokens"], target, pos)
        temp.append((inst, syn, score, margin))
        margins.append(margin)

    if not margins:
        return [], instances

    thr = np.percentile(margins, margin_percentile)

    seeds = []
    unlabeled = []
    for inst, syn, score, margin in temp:
        if syn is not None and margin >= thr:
            inst_copy = inst.copy()
            inst_copy["sense"] = syn
            inst_copy["margin"] = margin
            seeds.append(inst_copy)
        else:
            unlabeled.append(inst)
    return seeds, unlabeled


# -----------------------------
# 4. Context features
# -----------------------------
WINDOW = 5  # left/right window size


def context_string(inst):
    tokens = inst["tokens"]
    i = inst["target_index"]
    left = tokens[max(0, i - WINDOW) : i]
    right = tokens[i + 1 : i + 1 + WINDOW]
    ctx = left + right
    return " ".join(w.lower() for w in ctx)


# -----------------------------
# 5. Naive Bayes + Yarowsky-style bootstrapping
# -----------------------------
def classify_unlabeled(unlabeled, vectorizer, clf, conf_thr=0.7):
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


def yarowsky_bootstrap(seeds, unlabeled, max_iter=5, conf_thr=0.7, min_new=1):
    labeled = seeds[:]
    unl = unlabeled[:]

    if not labeled:
        return labeled, unl, None, None

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
        print(f"  Iter {it}: new_labeled = {len(new_labeled)}, remaining = {len(unl)}")

        if len(new_labeled) < min_new:
            print("  Stopping: too few new high-confidence labels.")
            break

        labeled.extend(new_labeled)
        X = vectorizer.fit_transform([context_string(inst) for inst in labeled])
        y = [
            inst["sense"].name() if hasattr(inst["sense"], "name") else inst["sense"]
            for inst in labeled
        ]

    return labeled, unl, vectorizer, clf


# -----------------------------
# 6. Pretty printing results per word
# -----------------------------
def pretty_print_results(word, labeled):
    print(f"\n=== Final labeled instances for '{word}' ===")
    from nltk.corpus import wordnet as wn

    for inst in labeled:
        doc_id = inst["doc_id"]
        sent = inst["sentence"]
        sense_val = inst["sense"]

        if hasattr(sense_val, "definition"):  # Synset object
            syn = sense_val
            sense_name = syn.name()
            gloss = syn.definition()
        else:  # string label, e.g. 'bank.n.09'
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


# -----------------------------
# 7. Run pipeline for multiple words and print summary table
# -----------------------------
def run_for_word(word, max_sent_per_word=10, margin_percentile=90, conf_thr=0.7):
    print(f"\n########## WORD = '{word}' ##########")
    corpus_sentences = get_semcor_sentences_for_word(word, max_sent=max_sent_per_word)
    print(f"Found {len(corpus_sentences)} SemCor sentences containing '{word}'.")

    if not corpus_sentences:
        return {
            "word": word,
            "total_inst": 0,
            "seeds": 0,
            "final_labeled": 0,
            "remaining_unlabeled": 0,
        }

    instances = extract_instances(corpus_sentences, word)
    print(f"Total instances of '{word}' in selected sentences: {len(instances)}")

    seeds, unlabeled = generate_seeds(
        instances, word, pos="n", margin_percentile=margin_percentile
    )
    print(f"Seeds: {len(seeds)}  Unlabeled (after Lesk): {len(unlabeled)}")

    final_labeled, final_unlabeled, vectorizer, clf = yarowsky_bootstrap(
        seeds, unlabeled, max_iter=5, conf_thr=conf_thr, min_new=1
    )

    summary = {
        "word": word,
        "total_inst": len(instances),
        "seeds": len(seeds),
        "final_labeled": len(final_labeled),
        "remaining_unlabeled": len(final_unlabeled),
    }

    # Optional detailed print:
    pretty_print_results(word, final_labeled)

    return summary


def print_summary_table(summaries):
    print("\n================ SUMMARY TABLE ================")
    header = f"{'Word':<10} {'Total':>5} {'Seeds':>7} {'FinalLbl':>9} {'Unlabeled':>10}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['word']:<10} "
            f"{s['total_inst']:>5} "
            f"{s['seeds']:>7} "
            f"{s['final_labeled']:>9} "
            f"{s['remaining_unlabeled']:>10}"
        )
    print("===============================================")


if __name__ == "__main__":
    # 10 polysemous English words (mostly nouns; you can tweak)
    target_words = [
        "bank",
        "plant",
        "interest",
        "cold",
        "operation",
        "charge",
        "head",
        "line",
        "light",
        "letter",
    ]

    summaries = []
    for w in target_words:
        summary = run_for_word(
            w,
            max_sent_per_word=10,   # ~10 sentences per word
            margin_percentile=90,   # strict Lesk seeding
            conf_thr=0.7,           # NB confidence threshold
        )
        summaries.append(summary)

    print_summary_table(summaries)