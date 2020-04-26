import numpy as np


class WRMSSEMetric:
    def __init__(self):
        pass

    def get_error(self, yhat, y, scale, weight):
        error = (1/12) * np.sum(
            weight * np.sqrt(((y - yhat) ** 2).mean(1) / scale))
        return error
