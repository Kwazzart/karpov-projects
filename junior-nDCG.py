from typing import List

import numpy as np


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if method not in ["standard", "industry"]:
        raise ValueError
    
    ideal_relevance = np.sort(relevance)[::-1]
    IDCG = 0
    DCG = 0
    if method == "standard":
        for i in range(k):
            IDCG += ideal_relevance[i] / np.log2(2 + i)
            DCG += relevance[i] / np.log2(2 + i)
    elif method == "industry":
        for i in range(k):
            IDCG += (2 ** ideal_relevance[i] - 1) / np.log2(2 + i)
            DCG += (2 ** relevance[i] - 1) / np.log2(2 + i)
    score = DCG/IDCG
    return score

