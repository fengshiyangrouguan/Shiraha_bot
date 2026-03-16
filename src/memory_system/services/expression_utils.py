import difflib
import random
import re
from typing import Dict, List, Optional


def filter_message_content(content: Optional[str]) -> str:
    if not content:
        return ""

    content = re.sub(r"\[回复.*?\]\s*", "", content)
    content = re.sub(r"@<[^>]*>", "", content)
    content = re.sub(r"\[picid:[^\]]*\]", "", content)
    content = re.sub(r"\[图片[^\]]*\]", "", content)
    return content.strip()


def calculate_similarity(text1: str, text2: str) -> float:
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def _compute_weights(population: List[Dict]) -> List[float]:
    if not population:
        return []

    counts: List[float] = []
    for item in population:
        count = item.get("count", 1)
        try:
            counts.append(max(float(count), 0.0))
        except (TypeError, ValueError):
            counts.append(1.0)

    min_count = min(counts)
    max_count = max(counts)
    if max_count == min_count:
        return [1.0 for _ in counts]

    weights: List[float] = []
    for count in counts:
        normalized = (count - min_count) / (max_count - min_count)
        weights.append(1.0 + normalized * 2.0)
    return weights


def weighted_sample(population: List[Dict], k: int) -> List[Dict]:
    if not population or k <= 0:
        return []

    if len(population) <= k:
        return population.copy()

    selected: List[Dict] = []
    population_copy = population.copy()

    for _ in range(min(k, len(population_copy))):
        weights = _compute_weights(population_copy)
        total_weight = sum(weights)
        if total_weight <= 0:
            idx = random.randint(0, len(population_copy) - 1)
            selected.append(population_copy.pop(idx))
            continue

        threshold = random.uniform(0, total_weight)
        cumulative = 0.0
        for idx, weight in enumerate(weights):
            cumulative += weight
            if threshold <= cumulative:
                selected.append(population_copy.pop(idx))
                break

    return selected
