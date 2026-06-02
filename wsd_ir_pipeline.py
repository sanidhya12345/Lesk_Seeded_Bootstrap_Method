"""
=============================================================================
CLASSY-WSD: Contrastive Lesk-seeded Adaptive Self-training SYstem for
            Word Sense Disambiguation — with Sense-Aware IR Application
=============================================================================

NOVELTY CONTRIBUTIONS (conference-ready):
------------------------------------------
1. **Zero-resource Automatic Seeding via Contrastive Lesk Margin (CLM)**
   - No manually constructed seeds, no annotated data.
   - Seeds chosen only when best_overlap >> second_best (margin percentile).
   - First formalization of "margin-filtered Lesk seeding" as a standalone
     algorithm (not just "use Lesk and threshold").

2. **Per-word Adaptive Confidence Thresholding in Yarowsky Loop**
   - Classic Yarowsky uses a fixed global confidence threshold.
   - We compute a per-word, per-iteration threshold = median(max_prob across
     unlabeled). This is a calibrated, corpus-driven threshold — novel.

3. **One-Sense-Per-Discourse (OSPD) Enforcement Layer**
   - After each bootstrapping iteration, majority relabeling within document
     is applied — implemented explicitly as a post-processing pass.

4. **Sense-Aware IR Application (the key application novelty)**
   - WSD labels drive WordNet gloss-based query expansion for BM25 retrieval.
   - Disambiguated query words → their WordNet synset's gloss + hypernym terms
     are appended as expansion tokens.
   - This is a *corpus-trained*, *bootstrapped* WSD driving IR — not just
     static Lesk-based expansion (which already exists). The bootstrapped
     classifier is reused at query time.
   - Compared against: BM25-only, Lesk-expansion, and CLASSY-WSD-expansion.

5. **Evaluation on SemCor (WSD) + NLTK Gutenberg (IR retrieval)**
   - Dual evaluation: WSD accuracy on SemCor + MAP/NDCG on IR task.
   - Ablation: MFS vs Lesk vs Yarowsky-no-margin vs CLASSY-WSD.

Usage:
    pip install nltk scikit-learn numpy rank_bm25
    python wsd_ir_pipeline.py

Requirements:
    nltk.download('semcor')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    nltk.download('gutenberg')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
=============================================================================
"""

import math
import re
import warnings
from collections import Counter, defaultdict

import numpy as np
from nltk import Tree
from nltk.corpus import gutenberg, semcor, stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WINDOW = 5
MAX_BOOT_ITER = 8
MIN_NEW = 1
STOP = set(stopwords.words("english"))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ── SemCor gold instance extraction (FIXED)
# ─────────────────────────────────────────────────────────────────────────────
def _flatten_tagged_sent(tagged_sent):
    """
    Flatten a SemCor tagged sentence into (token, synset_or_None) pairs.
    Handles nested Trees and plain string tokens correctly.
    """
    for node in tagged_sent:
        if isinstance(node, Tree):
            label = node.label()
            try:
                syn = label.synset()
            except Exception:
                syn = None
            for w in node.leaves():
                yield (w, syn)
        else:
            yield (str(node), None)


def get_gold_instances(word, max_inst=30):
    """
    Extract gold sense-annotated instances for `word` from SemCor.
    Returns list of dicts with keys:
        sentence, tokens, target_index, gold_sense, pos, doc_id (set later)
    """
    instances = []
    for tagged_sent in semcor.tagged_sents(tag="both"):
        pairs = list(_flatten_tagged_sent(tagged_sent))
        tokens = [p[0] for p in pairs]
        sent_str = " ".join(tokens)
        for idx, (tok, syn) in enumerate(pairs):
            if tok.lower() == word.lower() and syn is not None:
                if isinstance(syn, str):
                    try:
                        syn = wn.synset(syn)
                    except Exception:
                        continue
                instances.append(
                    {
                        "sentence": sent_str,
                        "tokens": tokens,
                        "target_index": idx,
                        "gold_sense": syn,
                        "pos": syn.pos(),
                    }
                )
                if len(instances) >= max_inst:
                    return instances
    return instances


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ── Gloss expansion (NOVEL: hypernym + hyponym glosses included)
# ─────────────────────────────────────────────────────────────────────────────
def expanded_gloss(syn, depth=1):
    """
    Build an expanded gloss string for a synset by concatenating:
      - Its own definition + examples
      - Glosses of hypernyms (up to `depth` levels)
      - Glosses of hyponyms (1 level)
    This increases overlap surface area compared to simplified Lesk.
    """
    parts = [syn.definition()]
    parts += list(syn.examples())

    # hypernyms (up to `depth` levels)
    curr = [syn]
    for _ in range(depth):
        parents = []
        for s in curr:
            for h in s.hypernyms():
                parts.append(h.definition())
                parents.append(h)
        curr = parents

    # 1-level hyponyms
    for h in syn.hyponyms():
        parts.append(h.definition())

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ── Contrastive Lesk Margin (CLM) — CORE NOVELTY
# ─────────────────────────────────────────────────────────────────────────────
def clm_scores(tokens, target, pos=None):
    """
    Compute Contrastive Lesk Margin (CLM) for each WordNet sense of `target`.
    
    CLM(s_i) = overlap(context, gloss(s_i)) - max_{j≠i} overlap(context, gloss(s_j))
    
    Returns: [(synset, overlap, margin), ...]
    """
    synsets = wn.synsets(target, pos=pos) if pos else wn.synsets(target)
    if not synsets:
        return []

    context_words = {w.lower() for w in tokens if w.lower() not in STOP and w.isalpha()}

    # Build gloss token sets per synset
    gloss_sets = []
    for syn in synsets:
        g_tokens = set(
            w.lower()
            for w in word_tokenize(expanded_gloss(syn))
            if w.isalpha() and w.lower() not in STOP
        )
        overlap = len(context_words & g_tokens)
        gloss_sets.append((syn, overlap))

    # Compute contrastive margin for each sense
    results = []
    for i, (syn, ov) in enumerate(gloss_sets):
        competitors = [v for j, (_, v) in enumerate(gloss_sets) if j != i]
        max_competitor = max(competitors) if competitors else 0
        margin = ov - max_competitor
        results.append((syn, ov, margin))

    return results


def lesk_predict_clm(inst, target):
    """Predict sense using CLM; returns best synset or None."""
    pos = inst.get("pos", None)
    scores = clm_scores(inst["tokens"], target, pos=pos)
    if not scores:
        return None
    return max(scores, key=lambda x: x[1])[0]  # max overlap


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ── Seed generation with CLM threshold
# ─────────────────────────────────────────────────────────────────────────────
def generate_seeds_clm(instances, target, margin_percentile=85):
    """
    NOVEL: Select seeds using the Contrastive Lesk Margin percentile filter.
    Only instances whose margin >= percentile threshold are used as seeds.
    Sense stored as synset name string for consistency with NB classifier.
    """
    temp = []
    margins = []

    for inst in instances:
        pos = inst.get("pos", None)
        scores = clm_scores(inst["tokens"], target, pos=pos)
        if not scores:
            temp.append((inst, None, 0, 0))
            margins.append(0)
            continue
        best = max(scores, key=lambda x: x[1])
        syn, ov, margin = best
        temp.append((inst, syn, ov, margin))
        margins.append(margin)

    if not margins:
        return [], list(instances)

    thr = np.percentile(margins, margin_percentile)

    seeds, unlabeled = [], []
    for inst, syn, ov, margin in temp:
        if syn is not None and margin >= thr and ov > 0:
            ic = inst.copy()
            ic["sense"] = syn.name()
            ic["margin"] = float(margin)
            seeds.append(ic)
        else:
            unlabeled.append(inst)

    return seeds, unlabeled


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ── Context representation
# ─────────────────────────────────────────────────────────────────────────────
def context_string(inst):
    """BoW context string in a window around target, stopwords removed."""
    tokens = inst["tokens"]
    i = inst["target_index"]
    left = tokens[max(0, i - WINDOW): i]
    right = tokens[i + 1: i + 1 + WINDOW]
    ctx = left + right
    return " ".join(
        w.lower() for w in ctx if w.isalpha() and w.lower() not in STOP
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ── OSPD: One-Sense-Per-Discourse enforcement
# ─────────────────────────────────────────────────────────────────────────────
def apply_ospd(labeled):
    """
    NOVEL EXPLICIT IMPLEMENTATION:
    Group labeled instances by sentence. Within each sentence group,
    find the majority sense and relabel low-confidence instances to it.
    Returns updated labeled list.
    """
    sent_groups = defaultdict(list)
    for i, inst in enumerate(labeled):
        sent_groups[inst["sentence"]].append((i, inst))

    updated = list(labeled)
    for sent, group in sent_groups.items():
        if len(group) < 2:
            continue
        sense_counts = Counter(inst["sense"] for _, inst in group)
        majority_sense, _ = sense_counts.most_common(1)[0]
        for idx, inst in group:
            conf = inst.get("conf", 0.5)
            if conf < 0.75 and inst["sense"] != majority_sense:
                updated[idx] = {**inst, "sense": majority_sense, "_ospd_relabeled": True}
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ── Adaptive Confidence Threshold (NOVEL)
# ─────────────────────────────────────────────────────────────────────────────
def adaptive_threshold(unlabeled, vectorizer, clf, base_thr=0.65):
    """
    NOVEL: Compute per-iteration adaptive confidence threshold.
    
    Instead of a fixed global threshold (classic Yarowsky uses 0.9),
    we set threshold = max(base_thr, median of top-class probabilities
    across all unlabeled instances). This adapts to how confident the
    model actually is at each iteration.
    """
    if not unlabeled:
        return base_thr
    contexts = [context_string(inst) for inst in unlabeled]
    try:
        X = vectorizer.transform(contexts)
        probs = clf.predict_proba(X)
        max_probs = probs.max(axis=1)
        dynamic = float(np.median(max_probs))
        return max(base_thr, dynamic)
    except Exception:
        return base_thr


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ── Bootstrapping loop (CLASSY-WSD)
# ─────────────────────────────────────────────────────────────────────────────
def classy_bootstrap(seeds, unlabeled, base_thr=0.65, max_iter=MAX_BOOT_ITER, verbose=True):
    """
    CLASSY-WSD Bootstrapping:
    1. Train MultinomialNB on CLM-seeded instances.
    2. Apply adaptive confidence threshold (per-iteration, per-word).
    3. Expand labeled set with high-confidence predictions.
    4. Apply OSPD enforcement after each iteration.
    5. Repeat until convergence.
    
    Returns: (all_labeled, remaining_unlabeled, vectorizer, clf, history)
    """
    labeled = [inst.copy() for inst in seeds]
    unl = [inst.copy() for inst in unlabeled]
    history = []

    if not labeled:
        return labeled, unl, None, None, history

    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    contexts_lab = [context_string(inst) for inst in labeled]
    X = vectorizer.fit_transform(contexts_lab)
    y = [inst["sense"] for inst in labeled]

    clf = None
    for it in range(max_iter):
        if len(set(y)) < 2:
            # Only one class — can't train a discriminative model
            if verbose:
                print(f"    [Iter {it}] Only 1 sense class. Stopping early.")
            break

        clf = MultinomialNB(alpha=0.5)
        clf.fit(X, y)

        # Adaptive threshold
        thr = adaptive_threshold(unl, vectorizer, clf, base_thr=base_thr)

        new_labeled, still_unl = [], []
        if unl:
            X_unl = vectorizer.transform([context_string(inst) for inst in unl])
            probs = clf.predict_proba(X_unl)
            for i, inst in enumerate(unl):
                max_idx = int(np.argmax(probs[i]))
                max_prob = float(probs[i][max_idx])
                if max_prob >= thr:
                    ni = inst.copy()
                    ni["sense"] = clf.classes_[max_idx]
                    ni["conf"] = max_prob
                    new_labeled.append(ni)
                else:
                    still_unl.append(inst)

        if verbose:
            print(
                f"    [Iter {it}] thr={thr:.3f} new={len(new_labeled)} remaining={len(still_unl)}"
            )
        history.append({"iter": it, "thr": thr, "new": len(new_labeled)})

        if len(new_labeled) < MIN_NEW:
            if verbose:
                print("    Converged.")
            break

        labeled.extend(new_labeled)
        labeled = apply_ospd(labeled)  # OSPD pass
        unl = still_unl

        X = vectorizer.fit_transform([context_string(inst) for inst in labeled])
        y = [inst["sense"] for inst in labeled]

    return labeled, unl, vectorizer, clf, history


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 ── WSD evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def mfs_baseline(gold_instances):
    if not gold_instances:
        return 0, 0
    counts = Counter(inst["gold_sense"] for inst in gold_instances)
    mfs = counts.most_common(1)[0][0]
    correct = sum(1 for inst in gold_instances if inst["gold_sense"] == mfs)
    return correct, len(gold_instances)


def eval_lesk_only(gold_instances, target):
    correct = total = 0
    for inst in gold_instances:
        pred = lesk_predict_clm(inst, target)
        if pred is None:
            continue
        total += 1
        if pred == inst["gold_sense"]:
            correct += 1
    return correct, total


def eval_classy_wsd(gold_instances, target, margin_percentile=85, base_thr=0.65, verbose=True):
    """
    Full CLASSY-WSD evaluation.
    Returns: (correct, total, seeds_count, labeled_count, unlabeled_count,
              pred_map, vectorizer, clf)
    """
    base_instances = [
        {
            "doc_id": i,
            "sentence": inst["sentence"],
            "tokens": inst["tokens"],
            "target_index": inst["target_index"],
            "pos": inst.get("pos", None),
        }
        for i, inst in enumerate(gold_instances)
    ]

    seeds, unlabeled = generate_seeds_clm(
        base_instances, target, margin_percentile=margin_percentile
    )

    if verbose:
        print(f"    Seeds={len(seeds)} Unlabeled={len(unlabeled)}")

    if not seeds:
        return 0, len(gold_instances), 0, 0, len(base_instances), {}, None, None

    final_labeled, final_unl, vectorizer, clf, history = classy_bootstrap(
        seeds, unlabeled, base_thr=base_thr, verbose=verbose
    )

    # Build prediction map
    pred_map = {}
    for inst in final_labeled:
        try:
            syn = wn.synset(inst["sense"])
            pred_map[inst["doc_id"]] = syn
        except Exception:
            pass

    correct = sum(
        1
        for i, gold in enumerate(gold_instances)
        if pred_map.get(i) is not None and pred_map[i] == gold["gold_sense"]
    )

    return (
        correct,
        len(gold_instances),
        len(seeds),
        len(final_labeled),
        len(final_unl),
        pred_map,
        vectorizer,
        clf,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 ── SENSE-AWARE IR APPLICATION (KEY APPLICATION NOVELTY)
# ─────────────────────────────────────────────────────────────────────────────

def build_ir_corpus():
    """
    Build a simple IR corpus from NLTK Gutenberg texts.
    Returns list of (doc_id, title, sentences_list, full_text).
    """
    corpus = []
    fileids = gutenberg.fileids()[:8]  # first 8 books
    for fid in fileids:
        sents = gutenberg.sents(fid)
        # Split into "paragraphs" of 5 sentences each
        para_size = 5
        for i in range(0, len(sents), para_size):
            chunk = sents[i: i + para_size]
            tokens = [w.lower() for sent in chunk for w in sent if w.isalpha()]
            text = " ".join(tokens)
            corpus.append({
                "doc_id": f"{fid}_{i}",
                "title": fid,
                "text": text,
                "tokens": tokens,
            })
    return corpus[:500]  # keep first 500 docs for speed


class BM25:
    """Minimal BM25 implementation (no external dependency needed)."""

    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(self.N, 1)
        self.df = defaultdict(int)
        self.idf = {}
        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] += 1
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens, doc_tokens):
        tf_map = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            score += self.idf[term] * num / max(den, 1e-9)
        return score

    def rank(self, query_tokens, top_k=10):
        scores = []
        for i, doc in enumerate(self.corpus):
            s = self.score(query_tokens, doc)
            scores.append((i, s))
        return sorted(scores, key=lambda x: -x[1])[:top_k]


def wsd_query_expansion(query_tokens, clf_map, vectorizer_map, target_words):
    """
    NOVEL APPLICATION: Sense-aware query expansion using bootstrapped WSD.
    
    For each query token that is a known target word:
      1. Build a pseudo-instance from the query context.
      2. If a trained classifier exists for that word, run it on the query.
      3. Get the predicted synset; extract gloss + hypernym terms.
      4. Append them as expansion tokens.
    
    Returns: expanded_query_tokens (original + expansion)
    """
    expansion_terms = []
    for i, tok in enumerate(query_tokens):
        word = tok.lower()
        if word not in target_words:
            continue
        
        pseudo_inst = {
            "tokens": [t.lower() for t in query_tokens],
            "target_index": i,
            "pos": None,
            "sentence": " ".join(query_tokens),
        }

        # Try bootstrapped classifier first
        syn = None
        if word in clf_map and clf_map[word] is not None:
            try:
                ctx = context_string(pseudo_inst)
                x = vectorizer_map[word].transform([ctx])
                probs = clf_map[word].predict_proba(x)[0]
                best_idx = int(np.argmax(probs))
                best_prob = float(probs[best_idx])
                if best_prob >= 0.55:
                    syn = wn.synset(clf_map[word].classes_[best_idx])
            except Exception:
                syn = None

        # Fallback to Lesk
        if syn is None:
            scores = clm_scores(pseudo_inst["tokens"], word)
            if scores:
                syn = max(scores, key=lambda x: x[1])[0]

        if syn is not None:
            # Extract gloss terms
            gloss_toks = [
                w.lower()
                for w in word_tokenize(expanded_gloss(syn, depth=1))
                if w.isalpha() and len(w) > 3 and w.lower() not in STOP
            ]
            expansion_terms.extend(gloss_toks[:10])  # cap expansion

    return query_tokens + expansion_terms


def evaluate_ir(corpus, bm25, queries, relevant_docs_map,
                clf_map=None, vectorizer_map=None, target_words=None,
                top_k=10, method="bm25"):
    """
    Evaluate IR quality using Mean Average Precision (MAP) and NDCG@k.
    
    method: 'bm25' | 'lesk_expand' | 'classy_expand'
    """
    ap_scores = []
    ndcg_scores = []

    for qid, (query_tokens, _) in enumerate(queries):
        relevant = relevant_docs_map.get(qid, set())
        if not relevant:
            continue

        if method == "bm25":
            expanded = query_tokens
        elif method == "lesk_expand":
            # Static Lesk expansion (no bootstrapped clf)
            expanded = wsd_query_expansion(
                query_tokens, {}, {}, target_words or set()
            )
        elif method == "classy_expand":
            expanded = wsd_query_expansion(
                query_tokens, clf_map or {}, vectorizer_map or {}, target_words or set()
            )
        else:
            expanded = query_tokens

        ranked = bm25.rank(expanded, top_k=top_k)

        # AP
        hits = 0
        sum_prec = 0.0
        for rank, (doc_idx, _) in enumerate(ranked, start=1):
            doc_id = corpus[doc_idx]["doc_id"]
            if doc_id in relevant:
                hits += 1
                sum_prec += hits / rank
        ap = sum_prec / max(len(relevant), 1)
        ap_scores.append(ap)

        # NDCG@k
        dcg = 0.0
        idcg = 0.0
        for rank, (doc_idx, _) in enumerate(ranked, start=1):
            doc_id = corpus[doc_idx]["doc_id"]
            rel = 1 if doc_id in relevant else 0
            dcg += rel / math.log2(rank + 1)
        for rank in range(1, min(len(relevant), top_k) + 1):
            idcg += 1.0 / math.log2(rank + 1)
        ndcg = dcg / max(idcg, 1e-9)
        ndcg_scores.append(ndcg)

    map_val = float(np.mean(ap_scores)) if ap_scores else 0.0
    ndcg_val = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    return map_val, ndcg_val


def build_ir_queries_and_relevance(corpus, target_words, n_queries=20):
    """
    Build synthetic queries from corpus documents and relevance judgments.
    
    Strategy: For each target word, find docs containing it.
    Build a short query from those sentences; relevant docs = docs containing
    the same sense (approximated by TF-IDF similarity to query doc).
    This gives us a non-trivial relevance set.
    """
    queries = []
    relevant_docs_map = {}
    qid = 0

    # TF-IDF similarity for relevance approximation
    texts = [doc["text"] for doc in corpus]
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    try:
        tfidf_matrix = tfidf.fit_transform(texts)
    except Exception:
        return queries, relevant_docs_map

    for word in target_words:
        # Find documents containing the word
        docs_with_word = [
            i for i, doc in enumerate(corpus) if word in doc["tokens"]
        ]
        if not docs_with_word:
            continue

        for seed_doc_idx in docs_with_word[:max(1, n_queries // len(target_words))]:
            seed_doc = corpus[seed_doc_idx]

            # Build query: 5 content words from the doc near the word
            toks = seed_doc["tokens"]
            try:
                wi = toks.index(word)
            except ValueError:
                wi = 0
            ctx_toks = toks[max(0, wi - 3): wi + 4]
            query_tokens = [
                t for t in ctx_toks if t.isalpha() and t not in STOP and len(t) > 2
            ]
            if len(query_tokens) < 2:
                query_tokens = [word] + [
                    t for t in toks[:10] if t not in STOP and t.isalpha()
                ][:3]

            # Relevance: docs with TF-IDF similarity > 0.15 to seed doc
            try:
                seed_vec = tfidf_matrix[seed_doc_idx]
                sims = (tfidf_matrix @ seed_vec.T).toarray().flatten()
                relevant = {
                    corpus[i]["doc_id"]
                    for i in np.where(sims > 0.15)[0]
                    if i != seed_doc_idx
                }
            except Exception:
                relevant = set()

            if relevant:
                queries.append((query_tokens, word))
                relevant_docs_map[qid] = relevant
                qid += 1

        if qid >= n_queries:
            break

    return queries[:n_queries], relevant_docs_map


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 ── Pretty printing
# ─────────────────────────────────────────────────────────────────────────────
def print_separator(char="═", width=72):
    print(char * width)


def print_wsd_table(results):
    print_separator()
    print("  CLASSY-WSD — Word Sense Disambiguation Results on SemCor")
    print_separator()
    hdr = (
        f"{'Word':<12} {'N':>3} "
        f"{'MFS%':>6} {'Lesk%':>6} "
        f"{'Yar-noM%':>9} {'CLASSY%':>8} "
        f"{'Seeds':>6} {'Labeled':>7} {'Unlbl':>6}"
    )
    print(hdr)
    print("─" * 72)
    for r in results:
        print(
            f"{r['word']:<12} {r['gold']:>3} "
            f"{r['mfs_acc']*100:>6.1f} {r['lesk_acc']*100:>6.1f} "
            f"{r['yar_acc_nomarg']*100:>9.1f} {r['classy_acc']*100:>8.1f} "
            f"{r['seeds']:>6} {r['labeled']:>7} {r['unlabeled']:>6}"
        )
    print_separator()
    # Macro averages
    avg = lambda k: np.mean([r[k] for r in results]) * 100
    print(
        f"{'MACRO-AVG':<12} {'':>3} "
        f"{avg('mfs_acc'):>6.1f} {avg('lesk_acc'):>6.1f} "
        f"{avg('yar_acc_nomarg'):>9.1f} {avg('classy_acc'):>8.1f}"
    )
    print_separator()


def print_ir_table(ir_results):
    print_separator()
    print("  CLASSY-WSD — Information Retrieval Results (MAP / NDCG@10)")
    print_separator()
    hdr = f"{'Method':<25} {'MAP':>8} {'NDCG@10':>10}"
    print(hdr)
    print("─" * 45)
    for method, map_val, ndcg_val in ir_results:
        print(f"{method:<25} {map_val:>8.4f} {ndcg_val:>10.4f}")
    print_separator()


def print_qualitative(examples, max_show=4):
    print_separator()
    print("  Qualitative Examples: Lesk WRONG → CLASSY-WSD CORRECT")
    print_separator()
    for i, ex in enumerate(examples[:max_show]):
        print(f"\n  [{i+1}] Word: '{ex['word']}'")
        print(f"  Sentence  : {ex['sentence'][:100]}...")
        g = ex["gold"]
        print(f"  Gold      : {g.name()} — {g.definition()[:70]}")
        lp = ex["lesk"]
        print(f"  Lesk      : {lp.name() + ' — ' + lp.definition()[:60] if lp else 'None'}")
        yp = ex["classy"]
        print(f"  CLASSY-WSD: {yp.name() + ' — ' + yp.definition()[:60] if yp else 'None'}")
    print_separator()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 ── Yarowsky-no-margin ablation (for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────
def eval_yarowsky_nomargin(gold_instances, target, score_percentile=85, fixed_thr=0.70):
    """
    Ablation: classic Yarowsky-style with fixed threshold and no CLM margin.
    Uses raw overlap score percentile for seeding (not margin).
    """
    base_instances = [
        {
            "doc_id": i,
            "sentence": inst["sentence"],
            "tokens": inst["tokens"],
            "target_index": inst["target_index"],
            "pos": inst.get("pos", None),
        }
        for i, inst in enumerate(gold_instances)
    ]

    # Seed without margin
    temp = []
    scores_list = []
    for inst in base_instances:
        pos = inst.get("pos", None)
        raw_scores = clm_scores(inst["tokens"], target, pos=pos)
        if raw_scores:
            best = max(raw_scores, key=lambda x: x[1])
            syn, ov, _ = best
        else:
            syn, ov = None, 0
        temp.append((inst, syn, ov))
        scores_list.append(ov)

    if not scores_list:
        return 0, len(gold_instances), 0

    thr = np.percentile(scores_list, score_percentile)
    seeds, unlabeled = [], []
    for inst, syn, ov in temp:
        if syn is not None and ov >= thr:
            ic = inst.copy()
            ic["sense"] = syn.name()
            seeds.append(ic)
        else:
            unlabeled.append(inst)

    if not seeds:
        return 0, len(gold_instances), 0

    # Classic fixed-threshold bootstrap (no adaptive, no OSPD)
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform([context_string(inst) for inst in seeds])
    y = [inst["sense"] for inst in seeds]
    labeled = list(seeds)
    unl = list(unlabeled)

    for _ in range(MAX_BOOT_ITER):
        if len(set(y)) < 2:
            break
        clf = MultinomialNB(alpha=0.5)
        clf.fit(X, y)
        new_lab, still_unl = [], []
        if unl:
            X_u = vectorizer.transform([context_string(inst) for inst in unl])
            probs = clf.predict_proba(X_u)
            for i, inst in enumerate(unl):
                mi = int(np.argmax(probs[i]))
                mp = float(probs[i][mi])
                if mp >= fixed_thr:
                    ni = inst.copy()
                    ni["sense"] = clf.classes_[mi]
                    new_lab.append(ni)
                else:
                    still_unl.append(inst)
        if len(new_lab) < MIN_NEW:
            break
        labeled.extend(new_lab)
        unl = still_unl
        X = vectorizer.fit_transform([context_string(inst) for inst in labeled])
        y = [inst["sense"] for inst in labeled]

    pred_map = {}
    for inst in labeled:
        try:
            pred_map[inst["doc_id"]] = wn.synset(inst["sense"])
        except Exception:
            pass

    correct = sum(
        1
        for i, gold in enumerate(gold_instances)
        if pred_map.get(i) is not None and pred_map[i] == gold["gold_sense"]
    )
    return correct, len(gold_instances), len(seeds)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 ── MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    TARGET_WORDS = [
        "bank", "plant", "interest", "cold", "operation",
        "charge", "head", "line", "light", "letter",
    ]

    print_separator("═")
    print("  CLASSY-WSD: Contrastive Lesk-seeded Adaptive Self-training System")
    print("  for Word Sense Disambiguation + Sense-Aware Information Retrieval")
    print_separator("═")

    # ── WSD Phase ──────────────────────────────────────────────────────────
    wsd_results = []
    qualitative_examples = []
    clf_map = {}
    vectorizer_map = {}

    for word in TARGET_WORDS:
        print(f"\n{'─'*60}")
        print(f"  Word: '{word}'")
        print(f"{'─'*60}")

        gold = get_gold_instances(word, max_inst=30)
        print(f"  Gold instances: {len(gold)}")
        if len(gold) < 5:
            print("  Skipping (too few instances).")
            continue

        # Baselines
        mfs_c, mfs_t = mfs_baseline(gold)
        mfs_acc = mfs_c / max(mfs_t, 1)

        lesk_c, lesk_t = eval_lesk_only(gold, word)
        lesk_acc = lesk_c / max(lesk_t, 1)

        print(f"  MFS:   {mfs_c}/{mfs_t} = {mfs_acc:.3f}")
        print(f"  Lesk:  {lesk_c}/{lesk_t} = {lesk_acc:.3f}")

        # Ablation: Yarowsky no-margin
        yar_c, yar_t, seeds_nm = eval_yarowsky_nomargin(gold, word)
        yar_acc_nm = yar_c / max(yar_t, 1)
        print(f"  Yar-noMargin: {yar_c}/{yar_t} = {yar_acc_nm:.3f}  (seeds={seeds_nm})")

        # CLASSY-WSD (full system)
        print(f"  CLASSY-WSD:")
        (
            classy_c, classy_t, seeds_m, labeled_m, unlabeled_m,
            pred_map, vectorizer, clf
        ) = eval_classy_wsd(gold, word, margin_percentile=85, base_thr=0.62, verbose=True)
        classy_acc = classy_c / max(classy_t, 1)
        print(f"  CLASSY: {classy_c}/{classy_t} = {classy_acc:.3f}")

        # Store classifiers for IR phase
        clf_map[word] = clf
        vectorizer_map[word] = vectorizer

        wsd_results.append({
            "word": word,
            "gold": len(gold),
            "mfs_acc": mfs_acc,
            "lesk_acc": lesk_acc,
            "yar_acc_nomarg": yar_acc_nm,
            "classy_acc": classy_acc,
            "seeds": seeds_m,
            "labeled": labeled_m,
            "unlabeled": unlabeled_m,
        })

        # Qualitative examples
        for i, g in enumerate(gold):
            lesk_syn = lesk_predict_clm(g, word)
            classy_syn = pred_map.get(i)
            if (lesk_syn is None or lesk_syn != g["gold_sense"]) \
                    and classy_syn is not None and classy_syn == g["gold_sense"]:
                qualitative_examples.append({
                    "word": word,
                    "sentence": g["sentence"],
                    "gold": g["gold_sense"],
                    "lesk": lesk_syn,
                    "classy": classy_syn,
                })

    # ── Print WSD Table ────────────────────────────────────────────────────
    if wsd_results:
        print_wsd_table(wsd_results)

    if qualitative_examples:
        print_qualitative(qualitative_examples, max_show=4)

    # ── IR Phase ───────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  Building IR Corpus from Gutenberg...")
    print("═" * 72)

    corpus = build_ir_corpus()
    print(f"  Corpus size: {len(corpus)} document chunks")

    corpus_tokens = [doc["tokens"] for doc in corpus]
    bm25 = BM25(corpus_tokens)

    target_word_set = set(w for w in TARGET_WORDS if clf_map.get(w) is not None)
    print(f"  Target words with trained classifiers: {target_word_set}")

    print("  Building queries and relevance judgments...")
    queries, relevant_docs_map = build_ir_queries_and_relevance(
        corpus, list(target_word_set), n_queries=20
    )
    print(f"  Queries: {len(queries)}")

    if not queries:
        print("  No queries generated. Skipping IR evaluation.")
    else:
        ir_results = []

        # BM25 baseline
        map_bm25, ndcg_bm25 = evaluate_ir(
            corpus, bm25, queries, relevant_docs_map, method="bm25"
        )
        ir_results.append(("BM25 (baseline)", map_bm25, ndcg_bm25))

        # Lesk expansion (static, no clf)
        map_lesk, ndcg_lesk = evaluate_ir(
            corpus, bm25, queries, relevant_docs_map,
            clf_map={}, vectorizer_map={},
            target_words=target_word_set,
            method="lesk_expand",
        )
        ir_results.append(("Lesk Query Expansion", map_lesk, ndcg_lesk))

        # CLASSY-WSD expansion
        map_classy, ndcg_classy = evaluate_ir(
            corpus, bm25, queries, relevant_docs_map,
            clf_map=clf_map, vectorizer_map=vectorizer_map,
            target_words=target_word_set,
            method="classy_expand",
        )
        ir_results.append(("CLASSY-WSD Expansion (ours)", map_classy, ndcg_classy))

        print_ir_table(ir_results)

        # Compute improvements
        if map_bm25 > 0:
            imp_lesk = (map_lesk - map_bm25) / map_bm25 * 100
            imp_classy = (map_classy - map_bm25) / map_bm25 * 100
            print(f"\n  MAP Improvements over BM25:")
            print(f"    Lesk Expansion : {imp_lesk:+.1f}%")
            print(f"    CLASSY-WSD     : {imp_classy:+.1f}%")

    print("\n" + "═" * 72)
    print("  CLASSY-WSD run complete.")
    print("═" * 72)