"""Retrieval metrics against gold evidence chunk IDs."""
from collections import Counter


def retrieval_metrics(rankings, golds, ks=(1, 3, 5, 10)):
    """rankings: one ranked list of chunk IDs per question; golds: one collection of gold chunk IDs per question.

    hit@k       : at least one gold chunk in the top k
    ev_recall@k : fraction of the question's gold chunks found in the top k, averaged (the CV metric)
    mrr@10      : reciprocal rank of the first gold chunk within the top max(ks)
    """
    hit = {k: 0.0 for k in ks}
    rec = {k: 0.0 for k in ks}
    mrr = 0.0
    K = max(ks)
    for ranked, gold in zip(rankings, golds):
        gold = set(gold)
        ranks = [r for r, c in enumerate(list(ranked)[:K]) if c in gold]
        if ranks:
            mrr += 1.0 / (ranks[0] + 1)
        for k in ks:
            n = sum(1 for r in ranks if r < k)
            hit[k] += n > 0
            rec[k] += n / len(gold)
    n = len(golds)
    out = {f"hit@{k}": round(hit[k] / n, 4) for k in ks}
    out.update({f"ev_recall@{k}": round(rec[k] / n, 4) for k in ks})
    out[f"mrr@{K}"] = round(mrr / n, 4)
    return out


def rank1_section_share(rankings, section_of, n_questions):
    """Share of questions whose rank-1 chunk comes from each section type."""
    c = Counter(section_of[r[0]] for r in rankings if len(r))
    return {t: round(v / n_questions, 4) for t, v in c.items()}


def gold_section_share(golds, section_of):
    """Share of all gold evidence chunks that sit in each section type."""
    c = Counter(section_of[g] for gold in golds for g in gold if g in section_of)
    n = sum(c.values())
    return {t: round(v / n, 4) for t, v in c.items()}


def markdown_table(results, cols=("hit@1", "hit@5", "ev_recall@5", "ev_recall@10", "mrr@10")):
    lines = ["| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)
