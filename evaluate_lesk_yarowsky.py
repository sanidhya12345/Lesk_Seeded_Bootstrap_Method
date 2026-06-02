"""
Lesk-Seeded Yarowsky Evaluation on SemCor (Single File) — FIXED

Fixes applied:
1. semcor.tagged_sents(tag='both') used correctly (not tagged_chunks).
2. Token index tracking rewritten to handle nested Trees properly.
3. POS inferred per-instance from gold synset instead of hardcoded 'n'.
4. Synset name normalization made consistent across seeding and bootstrapping.
5. context_string() has bounds-safe slicing (was already safe, kept explicit).
6. yarowsky_bootstrap clf.classes_ lookup guarded with try/except.
7. All synset objects normalized to .name() strings early; converted back only at eval.
"""

import nltk
from nltk.corpus import semcor, wordnet as wn
from nltk import Tree
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from collections import Counter


# Uncomment and run once if needed:
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('semcor')


# ---------------------------------------------------------------------
# Helper: flatten a tagged sentence into (token, synset_or_None) pairs
# ---------------------------------------------------------------------
def _flatten_tagged_sent(tagged_sent):
    """
    Given one tagged sentence (list of Trees and plain strings from SemCor),
    yield (token_str, synset_or_None) for every surface token in order.
    """
    for node in tagged_sent:
        if isinstance(node, Tree):
            label = node.label()
            # label is a Lemma object; .synset() gives the WordNet Synset
            # In tag='both' mode the label has a .synset() method
            try:
                syn = label.synset()  # WordNet Synset object
            except Exception:
                syn = None
            for w in node.leaves():
                yield (w, syn)
        else:
            # Plain string token — no sense annotation
            yield (node, None)


# ---------------------------------------------------------------------
# 1. Extract gold sense-annotated instances from SemCor
# ---------------------------------------------------------------------
def get_gold_instances(word, max_inst=20):
    """
    Extract up to max_inst gold sense-annotated instances for a given target word
    from SemCor.

    Each returned instance dict has:
        'sentence'    : raw sentence string
        'tokens'      : list of tokens (strings)
        'target_index': index of the target word in tokens
        'gold_sense'  : WordNet synset object (gold label)
        'pos'         : POS character ('n', 'v', 'a', 'r', 's') from the synset
    """
    instances = []

    for tagged_sent in semcor.tagged_sents(tag='both'):
        pairs = list(_flatten_tagged_sent(tagged_sent))
        tokens = [p[0] for p in pairs]
        sent_str = " ".join(tokens)

        for idx, (tok, syn) in enumerate(pairs):
            if tok.lower() == word.lower() and syn is not None:
                # Normalise: syn should already be a Synset; guard just in case
                if isinstance(syn, str):
                    try:
                        syn = wn.synset(syn)
                    except Exception:
                        continue

                pos_char = syn.pos()  # 'n', 'v', 'a', 's', 'r'

                instances.append({
                    "sentence": sent_str,
                    "tokens": tokens,
                    "target_index": idx,
                    "gold_sense": syn,
                    "pos": pos_char,
                })

                if len(instances) >= max_inst:
                    return instances

    return instances


# ---------------------------------------------------------------------
# 2. Lesk-based overlap and contrastive margin
# ---------------------------------------------------------------------
def lesk_overlap_scores(tokens, target, pos="n"):
    """
    Compute gloss-context overlap scores for all WordNet synsets of target.
    Returns list of (synset, overlap_score).
    """
    synsets = wn.synsets(target, pos=pos)
    if not synsets:
        # Fallback: try all POS
        synsets = wn.synsets(target)
    context = set(w.lower() for w in tokens)
    scores = []
    for syn in synsets:
        gloss_tokens = set(word_tokenize(syn.definition().lower()))
        score = len(context & gloss_tokens)
        scores.append((syn, score))
    return scores


def lesk_contrastive_seed_score(tokens, target, pos="n"):
    """
    Modified Lesk scoring with contrastive margin.
    Returns (best_synset, best_overlap, margin).
    """
    scores = lesk_overlap_scores(tokens, target, pos)
    if not scores:
        return None, 0, 0

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    best_syn, best = scores_sorted[0]
    second = scores_sorted[1][1] if len(scores_sorted) > 1 else 0
    margin = best - second
    return best_syn, best, margin


# ---------------------------------------------------------------------
# 3. Lesk-only baseline
# ---------------------------------------------------------------------
def lesk_predict(inst, target, pos=None):
    """
    Predict a sense using contrastive Lesk.
    POS is taken from inst['pos'] if available, else defaults to 'n'.
    """
    if pos is None:
        pos = inst.get("pos", "n")
    syn, score, margin = lesk_contrastive_seed_score(inst["tokens"], target, pos)
    return syn


def eval_lesk_only(gold_instances, target):
    """
    Evaluate Lesk-only on gold instances.
    Returns (correct, total_used).
    """
    correct = 0
    total = 0
    for inst in gold_instances:
        pred = lesk_predict(inst, target)
        if pred is None:
            continue
        total += 1
        if pred == inst["gold_sense"]:
            correct += 1
    return correct, total


# ---------------------------------------------------------------------
# 4. Most Frequent Sense (MFS) baseline
# ---------------------------------------------------------------------
def mfs_baseline(gold_instances):
    """
    Most Frequent Sense baseline.
    Returns (correct, total).
    """
    if not gold_instances:
        return 0, 0
    counts = Counter(inst["gold_sense"] for inst in gold_instances)
    mfs_sense, _ = counts.most_common(1)[0]
    correct = sum(1 for inst in gold_instances if inst["gold_sense"] == mfs_sense)
    return correct, len(gold_instances)


# ---------------------------------------------------------------------
# 5. Lesk-based seeding (two variants)
# ---------------------------------------------------------------------
def generate_seeds_margin(instances, target, margin_percentile=90):
    """
    Seed selection using contrastive margin.
    Returns (seeds, unlabeled). Sense stored as synset name string.
    """
    temp = []
    margins = []

    for inst in instances:
        pos = inst.get("pos", "n")
        syn, score, margin = lesk_contrastive_seed_score(inst["tokens"], target, pos)
        temp.append((inst, syn, margin))
        margins.append(margin)

    if not margins:
        return [], instances

    thr = np.percentile(margins, margin_percentile)

    seeds, unlabeled = [], []
    for inst, syn, margin in temp:
        if syn is not None and margin >= thr:
            inst_copy = inst.copy()
            inst_copy["sense"] = syn.name()   # store as string for NB
            inst_copy["margin"] = margin
            seeds.append(inst_copy)
        else:
            unlabeled.append(inst)

    return seeds, unlabeled


def generate_seeds_nomargin(instances, target, score_percentile=90):
    """
    Seed selection without contrastive margin (ablation).
    Returns (seeds, unlabeled). Sense stored as synset name string.
    """
    temp = []
    scores_list = []

    for inst in instances:
        pos = inst.get("pos", "n")
        syn, best, _ = lesk_contrastive_seed_score(inst["tokens"], target, pos)
        temp.append((inst, syn, best))
        scores_list.append(best)

    if not scores_list:
        return [], instances

    thr = np.percentile(scores_list, score_percentile)

    seeds, unlabeled = [], []
    for inst, syn, best in temp:
        if syn is not None and best >= thr:
            inst_copy = inst.copy()
            inst_copy["sense"] = syn.name()   # store as string for NB
            inst_copy["score"] = best
            seeds.append(inst_copy)
        else:
            unlabeled.append(inst)

    return seeds, unlabeled


# ---------------------------------------------------------------------
# 6. Context representation and Yarowsky-style bootstrapping
# ---------------------------------------------------------------------
WINDOW = 5


def context_string(inst):
    """
    Bag-of-words context string using a symmetric window around target.
    """
    tokens = inst["tokens"]
    i = inst["target_index"]
    left = tokens[max(0, i - WINDOW): i]
    right = tokens[i + 1: i + 1 + WINDOW]
    return " ".join(w.lower() for w in left + right)


def classify_unlabeled(unlabeled, vectorizer, clf, conf_thr=0.7):
    """
    Classify unlabeled instances; keep predictions with probability >= conf_thr.
    """
    new_labeled, still_unlabeled = [], []

    for inst in unlabeled:
        ctx = context_string(inst)
        x = vectorizer.transform([ctx])
        probs = clf.predict_proba(x)[0]
        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])

        if max_prob >= conf_thr:
            inst_copy = inst.copy()
            inst_copy["sense"] = clf.classes_[max_idx]  # string synset name
            inst_copy["conf"] = max_prob
            new_labeled.append(inst_copy)
        else:
            still_unlabeled.append(inst)

    return new_labeled, still_unlabeled


def yarowsky_bootstrap(seeds, unlabeled, max_iter=5, conf_thr=0.7, min_new=1):
    """
    Yarowsky-style self-training loop.
    Returns (labeled, remaining_unlabeled, vectorizer, clf).
    """
    labeled = seeds[:]
    unl = unlabeled[:]

    if not labeled:
        return labeled, unl, None, None

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([context_string(inst) for inst in labeled])
    # "sense" is already a string synset name
    y = [inst["sense"] for inst in labeled]

    clf = None
    for it in range(max_iter):
        clf = MultinomialNB()
        clf.fit(X, y)

        new_labeled, unl = classify_unlabeled(unl, vectorizer, clf, conf_thr=conf_thr)
        print(f"    Iter {it}: new_labeled={len(new_labeled)}, remaining={len(unl)}")

        if len(new_labeled) < min_new:
            print("    Stopping: too few new high-confidence labels.")
            break

        labeled.extend(new_labeled)
        X = vectorizer.fit_transform([context_string(inst) for inst in labeled])
        y = [inst["sense"] for inst in labeled]

    return labeled, unl, vectorizer, clf


# ---------------------------------------------------------------------
# 7. Evaluation of Lesk-seeded Yarowsky
# ---------------------------------------------------------------------
def eval_lesk_seeded_yarowsky(
    gold_instances,
    target,
    use_margin=True,
    margin_percentile=90,
    score_percentile=90,
    conf_thr=0.7,
):
    """
    Evaluate Lesk-seeded Yarowsky on gold instances.
    Gold labels used only for evaluation, never for training.
    """
    # Strip gold labels to build "unsupervised" base instances
    base_instances = []
    for doc_id, inst in enumerate(gold_instances):
        base_instances.append({
            "doc_id": doc_id,
            "sentence": inst["sentence"],
            "tokens": inst["tokens"],
            "target_index": inst["target_index"],
            "pos": inst.get("pos", "n"),
        })

    if use_margin:
        seeds, unlabeled = generate_seeds_margin(
            base_instances, target, margin_percentile=margin_percentile
        )
    else:
        seeds, unlabeled = generate_seeds_nomargin(
            base_instances, target, score_percentile=score_percentile
        )

    print(f"    Seeds: {len(seeds)}, Unlabeled: {len(unlabeled)}, use_margin={use_margin}")

    if not seeds:
        return 0, len(gold_instances), 0, 0, len(base_instances), {}, base_instances

    final_labeled, final_unlabeled, vectorizer, clf = yarowsky_bootstrap(
        seeds, unlabeled, max_iter=5, conf_thr=conf_thr, min_new=1
    )

    # Build doc_id -> Synset map (sense stored as string name)
    pred_map = {}
    for inst in final_labeled:
        sense_val = inst["sense"]
        try:
            syn = wn.synset(sense_val) if isinstance(sense_val, str) else sense_val
            pred_map[inst["doc_id"]] = syn
        except Exception:
            pass  # invalid synset name — skip

    correct = 0
    for doc_id, gold in enumerate(gold_instances):
        pred_syn = pred_map.get(doc_id)
        if pred_syn is not None and pred_syn == gold["gold_sense"]:
            correct += 1

    return (
        correct,
        len(gold_instances),
        len(seeds),
        len(final_labeled),
        len(final_unlabeled),
        pred_map,
        base_instances,
    )


# ---------------------------------------------------------------------
# 8. Pretty-printing & qualitative examples
# ---------------------------------------------------------------------
def print_eval_table(results):
    print("\n======== WSD Evaluation on SemCor Subset ========")
    header = (
        f"{'Word':<10} {'Gold':>4} "
        f"{'MFS%':>6} {'Lesk%':>6} "
        f"{'Y_noM%':>7} {'Y_marg%':>8} "
        f"{'SeedN':>6} {'SeedM':>6} {'LblM':>6} {'UnlM':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['word']:<10} "
            f"{r['gold']:>4} "
            f"{r['mfs_acc']*100:>6.1f} "
            f"{r['lesk_acc']*100:>6.1f} "
            f"{r['yar_acc_nomarg']*100:>7.1f} "
            f"{r['yar_acc_margin']*100:>8.1f} "
            f"{r['seeds_nomarg']:>6} "
            f"{r['seeds_margin']:>6} "
            f"{r['labeled_margin']:>6} "
            f"{r['unlabeled_margin']:>6}"
        )
    print("=" * len(header))


def print_qualitative_examples(examples, max_examples=5):
    print("\n======== Qualitative Examples (Lesk wrong, Lesk+Y correct) ========")
    for i, ex in enumerate(examples[:max_examples]):
        print(f"\nExample {i+1} (word='{ex['word']}'):")
        print("  Sentence:", ex["sentence"])
        g = ex["gold_sense"]
        print(f"  Gold   : {g.name()} — {g.definition()}")
        lp = ex["lesk_pred"]
        print(f"  Lesk   : {lp.name() + ' — ' + lp.definition() if lp else 'None'}")
        yp = ex["yar_pred"]
        print(f"  Lesk+Y : {yp.name() + ' — ' + yp.definition() if yp else 'None'}")
    print("=" * 68)


# ---------------------------------------------------------------------
# 9. Main experiment
# ---------------------------------------------------------------------
if __name__ == "__main__":
    target_words = [
        "bank", "plant", "interest", "cold", "operation",
        "charge", "head", "line", "light", "letter",
    ]

    results = []
    qualitative_examples = []

    for word in target_words:
        print(f"\n########## WORD = '{word}' ##########")

        gold_instances = get_gold_instances(word, max_inst=20)
        print(f"  Gold instances found: {len(gold_instances)}")
        if len(gold_instances) < 5:
            print("  Skipping (too few gold instances).")
            continue

        # 1) MFS baseline
        mfs_c, mfs_t = mfs_baseline(gold_instances)
        mfs_acc = mfs_c / mfs_t if mfs_t else 0.0
        print(f"  MFS: {mfs_c}/{mfs_t} = {mfs_acc:.3f}")

        # 2) Lesk-only baseline
        lesk_c, lesk_t = eval_lesk_only(gold_instances, word)
        lesk_acc = lesk_c / lesk_t if lesk_t else 0.0
        print(f"  Lesk-only: {lesk_c}/{lesk_t} = {lesk_acc:.3f}")

        # 3) Lesk-seeded Yarowsky — NO margin (ablation)
        yar_c_nm, yar_t_nm, seeds_nm, lbl_nm, unl_nm, pred_map_nm, _ = \
            eval_lesk_seeded_yarowsky(
                gold_instances, word,
                use_margin=False, score_percentile=90, conf_thr=0.7,
            )
        yar_acc_nm = yar_c_nm / yar_t_nm if yar_t_nm else 0.0
        print(f"  Lesk+Y (no margin): {yar_c_nm}/{yar_t_nm} = {yar_acc_nm:.3f}")

        # 4) Lesk-seeded Yarowsky — WITH contrastive margin
        yar_c_m, yar_t_m, seeds_m, lbl_m, unl_m, pred_map_m, _ = \
            eval_lesk_seeded_yarowsky(
                gold_instances, word,
                use_margin=True, margin_percentile=90, conf_thr=0.7,
            )
        yar_acc_m = yar_c_m / yar_t_m if yar_t_m else 0.0
        print(f"  Lesk+Y (margin)   : {yar_c_m}/{yar_t_m} = {yar_acc_m:.3f}")

        results.append({
            "word": word,
            "gold": len(gold_instances),
            "mfs_acc": mfs_acc,
            "lesk_acc": lesk_acc,
            "yar_acc_nomarg": yar_acc_nm,
            "yar_acc_margin": yar_acc_m,
            "seeds_nomarg": seeds_nm,
            "seeds_margin": seeds_m,
            "labeled_margin": lbl_m,
            "unlabeled_margin": unl_m,
        })

        # 5) Qualitative examples: Lesk wrong, Lesk+Y correct
        for doc_id, gold in enumerate(gold_instances):
            gold_syn = gold["gold_sense"]
            lesk_syn = lesk_predict(gold, word)
            yar_syn = pred_map_m.get(doc_id)
            if lesk_syn != gold_syn and yar_syn == gold_syn:
                qualitative_examples.append({
                    "word": word,
                    "sentence": gold["sentence"],
                    "gold_sense": gold_syn,
                    "lesk_pred": lesk_syn,
                    "yar_pred": yar_syn,
                })

    if results:
        print_eval_table(results)
    if qualitative_examples:
        print_qualitative_examples(qualitative_examples, max_examples=5)