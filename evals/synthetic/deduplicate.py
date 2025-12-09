"""
Cross-sample deduplication using TF-IDF cosine similarity.

Removes near-duplicate responses across the entire dataset to prevent
repetitive patterns in training data.

Uses TF-IDF vectorization with cosine similarity threshold of 0.9.
At our scale (~1,600 responses), O(n²) pairwise comparison is fast enough.

Usage:
    # Deduplicate filtered responses
    python -m evals.synthetic.deduplicate --input filtered.jsonl

    # Use different threshold
    python -m evals.synthetic.deduplicate --input filtered.jsonl --threshold 0.85
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Lazy imports for sklearn (not needed for dry-run)
TfidfVectorizer = None
cosine_similarity = None
np = None


def _ensure_sklearn():
    """Lazily import sklearn dependencies."""
    global TfidfVectorizer, cosine_similarity, np
    if TfidfVectorizer is None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
            import numpy as _np
            TfidfVectorizer = _TfidfVectorizer
            cosine_similarity = _cosine_similarity
            np = _np
        except ImportError as e:
            raise ImportError(
                "sklearn is required for deduplication. Install with: pip install scikit-learn"
            ) from e

# Default threshold
DEFAULT_THRESHOLD = 0.90


def load_responses(input_path: Path) -> List[Dict]:
    """Load responses from JSONL."""
    responses = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return responses


def deduplicate_responses(
    responses: List[Dict],
    threshold: float = DEFAULT_THRESHOLD,
    verbose: bool = False,
) -> Tuple[List[Dict], List[Tuple[int, int, float]]]:
    """
    Remove near-duplicate responses using TF-IDF cosine similarity.

    Args:
        responses: List of response dictionaries
        threshold: Similarity threshold (0-1), responses above are duplicates
        verbose: Print duplicate pairs

    Returns:
        Tuple of (unique_responses, duplicate_pairs)
        where duplicate_pairs is list of (kept_idx, removed_idx, similarity)
    """
    _ensure_sklearn()  # Lazy import

    if not responses:
        return responses, []

    # Extract response texts
    texts = [r['response'] for r in responses]

    # Build TF-IDF matrix
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words='english',
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Track which responses to keep
    unique_indices = [0]  # Always keep first
    duplicate_pairs = []

    for i in range(1, len(responses)):
        # Compare against all kept responses
        sims = cosine_similarity(
            tfidf_matrix[i:i+1],
            tfidf_matrix[unique_indices]
        ).flatten()

        max_sim = sims.max()
        max_idx = unique_indices[sims.argmax()]

        if max_sim < threshold:
            # Sufficiently different, keep it
            unique_indices.append(i)
        else:
            # Too similar, mark as duplicate
            duplicate_pairs.append((max_idx, i, max_sim))
            if verbose:
                print(f"Duplicate: {i} ~ {max_idx} (sim={max_sim:.3f})")
                print(f"  Kept: {responses[max_idx]['sample_id']}/{responses[max_idx]['persona_name']}")
                print(f"  Removed: {responses[i]['sample_id']}/{responses[i]['persona_name']}")

    unique_responses = [responses[i] for i in unique_indices]
    return unique_responses, duplicate_pairs


def analyze_duplicates(
    responses: List[Dict],
    duplicate_pairs: List[Tuple[int, int, float]],
) -> Dict:
    """Analyze patterns in detected duplicates."""
    stats = {
        'total_duplicates': len(duplicate_pairs),
        'by_persona': defaultdict(int),
        'by_subject': defaultdict(int),
        'cross_sample': 0,
        'same_sample': 0,
        'similarity_distribution': [],
    }

    for kept_idx, removed_idx, sim in duplicate_pairs:
        kept = responses[kept_idx]
        removed = responses[removed_idx]

        stats['by_persona'][removed['persona_name']] += 1
        stats['by_subject'][removed['subject']] += 1
        stats['similarity_distribution'].append(sim)

        if kept['sample_id'] == removed['sample_id']:
            stats['same_sample'] += 1
        else:
            stats['cross_sample'] += 1

    return stats


def save_responses(responses: List[Dict], output_path: Path):
    """Save responses to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def save_duplicates_log(
    responses: List[Dict],
    duplicate_pairs: List[Tuple[int, int, float]],
    output_path: Path,
):
    """Save detailed duplicate log for analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for kept_idx, removed_idx, sim in duplicate_pairs:
            kept = responses[kept_idx]
            removed = responses[removed_idx]

            log_entry = {
                'similarity': sim,
                'kept': {
                    'sample_id': kept['sample_id'],
                    'persona_name': kept['persona_name'],
                    'subject': kept['subject'],
                    'weighted_score': kept['weighted_score'],
                    'response_preview': kept['response'][:200] + '...',
                },
                'removed': {
                    'sample_id': removed['sample_id'],
                    'persona_name': removed['persona_name'],
                    'subject': removed['subject'],
                    'weighted_score': removed['weighted_score'],
                    'response_preview': removed['response'][:200] + '...',
                },
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


def print_stats(
    n_input: int,
    n_output: int,
    stats: Dict,
):
    """Print deduplication statistics."""
    _ensure_sklearn()  # For np.mean/median

    print(f"\n{'='*60}")
    print("DEDUPLICATION STATISTICS")
    print(f"{'='*60}")

    print(f"\n📊 Overall:")
    print(f"  Input responses:   {n_input}")
    print(f"  Output responses:  {n_output}")
    print(f"  Duplicates found:  {stats['total_duplicates']} ({100*stats['total_duplicates']/n_input:.1f}%)")

    if stats['total_duplicates'] > 0:
        print(f"\n🔍 Duplicate Types:")
        print(f"  Same sample:     {stats['same_sample']}")
        print(f"  Cross sample:    {stats['cross_sample']}")

        print(f"\n🎭 By Persona (removed):")
        for persona, count in sorted(stats['by_persona'].items(), key=lambda x: -x[1]):
            print(f"  {persona:25s}: {count}")

        print(f"\n📚 By Subject (removed):")
        for subject, count in sorted(stats['by_subject'].items(), key=lambda x: -x[1]):
            print(f"  {subject:20s}: {count}")

        sims = stats['similarity_distribution']
        print(f"\n📈 Similarity Distribution:")
        print(f"  Min:    {min(sims):.3f}")
        print(f"  Max:    {max(sims):.3f}")
        print(f"  Mean:   {np.mean(sims):.3f}")
        print(f"  Median: {np.median(sims):.3f}")

    print(f"{'='*60}\n")


def run_deduplication(
    input_path: Path,
    output_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    save_log: bool = False,
    verbose: bool = False,
):
    """Run the full deduplication pipeline."""

    print(f"\n{'='*60}")
    print("CROSS-SAMPLE DEDUPLICATION")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Similarity threshold: {threshold}")
    print(f"{'='*60}\n")

    # Load responses
    print("Loading responses...")
    responses = load_responses(input_path)
    print(f"Loaded {len(responses)} responses")

    # Deduplicate
    print(f"\nFinding duplicates (threshold={threshold})...")
    unique_responses, duplicate_pairs = deduplicate_responses(
        responses, threshold, verbose
    )
    print(f"Found {len(duplicate_pairs)} duplicates")
    print(f"Keeping {len(unique_responses)} unique responses")

    # Analyze duplicates
    stats = analyze_duplicates(responses, duplicate_pairs)

    # Save results
    save_responses(unique_responses, output_path)
    print(f"\n✅ Saved {len(unique_responses)} unique responses to {output_path}")

    if save_log and duplicate_pairs:
        log_path = input_path.parent / f"{input_path.stem}_duplicates_log.jsonl"
        save_duplicates_log(responses, duplicate_pairs, log_path)
        print(f"📝 Saved duplicate log to {log_path}")

    # Print stats
    print_stats(len(responses), len(unique_responses), stats)

    return unique_responses


def main():
    parser = argparse.ArgumentParser(description="Deduplicate synthetic responses")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with filtered responses")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: input with _deduped suffix)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--save-log", action="store_true",
                        help="Save detailed duplicate log")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each duplicate pair found")

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_deduped.jsonl"

    run_deduplication(
        input_path=input_path,
        output_path=output_path,
        threshold=args.threshold,
        save_log=args.save_log,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
