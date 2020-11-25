
"""
Utilities to keep track of averages 

Author: Nishant Prabhu

"""


class Scalar:
    """
    Class to keep track of a variable's values
    """
    def __init__(self):
        self.sum = 0
        self.n_updates = 0
        self.mean = 0
        self.last_value = None

    def update(self, x):
        self.last_value = x
        self.sum += x
        self.n_updates += 1 
        self.mean = self.sum/self.n_updates