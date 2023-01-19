import numpy as np

def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    '''Calculate LTV Metric'''
    errors = np.sqrt(((y_true - y_pred) / (y_pred ** -1)) ** 2)
    error = float(np.mean(errors))
    
    return error
