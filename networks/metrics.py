from typing import List, Tuple


# Levenshtein distance between two sequences (a, b) at element level
def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


# Compute the Symbol Error Rate (%) and the Sequence Error Rate (%)
# over a pair of ground-truth and predictions labels
# Labels are nested string lists with no padding
def compute_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Tuple[float, float]:
    ed_acc = 0
    length_acc = 0
    label_acc = 0
    counter = 0

    for t, h in zip(y_true, y_pred):
        ed = levenshtein(t, h)
        ed_acc += ed
        length_acc += len(t)
        if ed > 0:
            label_acc += 1
        counter += 1

    symer = 100.0 * ed_acc / length_acc
    seqer = 100.0 * label_acc / counter

    return symer, seqer
