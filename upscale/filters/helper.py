"""
This module contains some simple-stupid helper filters.
"""

class OutputAllFilter:
    """This filter copies all input signals to the output signals.
    The list of propagated signals will be empty.
    """
    
    def apply_filter(self, signal):
        return {'prop': [], 'out': [signal]}
