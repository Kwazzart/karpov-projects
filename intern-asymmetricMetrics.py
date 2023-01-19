import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """Calculate asymmetric metric"""
    errors = (((y_true - y_pred) / y_pred) ** 2)
    mean_error = np.mean(errors)
    floated_error = float(mean_error)
    
    return floated_error