import json
from collections import Counter
from pathlib import Path

import numpy as np
from more_itertools import batched
from reach import Reach
from tokenizers import Tokenizer
from tqdm import tqdm


def collect_means_and_texts(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    """Collect means and texts from a list of reach paths."""
    txts = []
    v = []
    for path in tqdm(paths, desc="Collecting means and texts"):
        if not path.name.endswith(".json"):
            continue
        try:
            r = Reach.load(path)
        except KeyError:
            # Workaround for old format reach
            vectors_path = str(path).replace("_items.json", "_vectors.npy")
            items = json.load(open(path))["items"]
            vectors = np.load(open(vectors_path, "rb"))
            r = Reach(vectors, items)
        # Filter out any NaN vectors before appending
        non_nan_indices = ~np.isnan(r.vectors).any(axis=1)
        valid_vectors = r.vectors[non_nan_indices]
        valid_items = np.array(r.sorted_items)[non_nan_indices]
        txts.extend(valid_items)
        v.append(valid_vectors)

    return txts, np.concatenate(v)


def calculate_token_probabilities(tokenizer: Tokenizer, txt: list[str]) -> np.ndarray:
    """Count tokens in a set of texts."""
    vocab_size = tokenizer.get_vocab_size()
    counts: Counter[int] = Counter()
    for t in tqdm(batched(txt, 1024)):
        encodings = tokenizer.encode_batch_fast(t, add_special_tokens=False)
        for e in encodings:
            counts.update(e.ids)

    # Add the number of tokens to account for smoothing
    sum_id = sum(counts.values()) + vocab_size
    # Start with ones for smoothing
    x = np.ones(vocab_size)

    for word_id, count in counts.items():
        x[word_id] += count

    # Turn into probabilities
    x /= sum_id

    return x
