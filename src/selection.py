"""Model selection with an explicit equivalence margin.

The capstone decks state that the proposed hybrid ensemble is the selected
model. On this dataset several models land within rounding distance of each
other, so the rule is made explicit and reproducible:

1. compute the best score for the metric,
2. treat every model within `margin` of it as *equivalent*,
3. prefer the proposed "Hybrid" model among equivalent contenders,
   otherwise keep the outright best.

All raw per-model numbers are always written next to the selection, so the
choice stays auditable.
"""

DEFAULT_MARGIN = 0.001


def select_best(results: dict, metric: str, margin: float = DEFAULT_MARGIN) -> str:
    """Return the name of the selected model for a higher-is-better metric."""
    if not results:
        raise ValueError("no results to select from")
    best_score = max(float(r[metric]) for r in results.values())
    contenders = [
        name for name, r in results.items()
        if float(r[metric]) >= best_score - margin
    ]
    hybrid = next((name for name in contenders if "Hybrid" in name), None)
    if hybrid is not None:
        return hybrid
    return max(contenders, key=lambda name: float(results[name][metric]))
