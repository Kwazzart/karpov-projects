from typing import List

import numpy as np

def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
    """avarage nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values ​​\
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if method not in ["standard", "industry"]:
        raise ValueError
    
    nDCG_list = []
    for query in list_relevances:
        best_query = sorted(query)[::-1]
        IDCG = 0
        DCG = 0
        if method == "standard":
            for i in range(k):
                IDCG += best_query[i] / np.log2(2 + i)
                DCG += query[i] / np.log2( 2 + i)
        elif method == "industry":
            for i in range(k):
                IDCG += (2 ** best_query[i] - 1) / np.log2(2 + i)
                DCG += (2 ** query[i] - 1) / np.log2( 2 + i)
        nDCG = DCG/IDCG
        nDCG_list.append(nDCG)
    score = np.mean(nDCG_list)
    return score