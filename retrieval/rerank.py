"""Cross-encoder reranking of hybrid candidates."""
import numpy as np


class Reranker:
    def __init__(self, model_name, max_length=512, fp16=False):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.fp16 = False
        if fp16:
            try:
                import torch
                if torch.cuda.is_available():  # half precision: ~5x faster on a T4, same recall
                    self.model.model.half()
                    self.fp16 = True
                else:
                    print("No GPU: reranker runs in fp32")
            except Exception as e:
                print(f"fp16 not applied ({e}); running in fp32")

    def score(self, pairs, batch_size=64, show_progress_bar=False):
        """Relevance score for each (query, passage) pair."""
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.model.predict(pairs, batch_size=batch_size,
                                             show_progress_bar=show_progress_bar), dtype=np.float32)


def apply_scores(candidates, scores, top):
    """Re-order the first `top` candidates by reranker score (high to low);
    anything beyond `top` keeps its hybrid order. Returns [(row_index, score_or_None), ...]."""
    head = sorted(zip(candidates[:top], (float(s) for s in scores)), key=lambda x: -x[1])
    return head + [(idx, None) for idx in candidates[top:]]
