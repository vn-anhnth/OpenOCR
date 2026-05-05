import sys
import os
import numpy as np

# Mocking parts of the environment if needed
sys.path.append(os.getcwd())

from openrec.metrics.rec_metric import RecMetric

def test_cer():
    metric = RecMetric()
    
    # Test case 1: Exact match
    preds = [('hello', 0.9)]
    labels = [('hello', None)]
    res = metric((preds, labels))
    print(f"Test 1 (Exact): {res}")
    # CER should be 0.0
    
    metric.reset()
    
    # Test case 2: Single substitution
    preds = [('hella', 0.9)]
    labels = [('hello', None)]
    res = metric((preds, labels))
    print(f"Test 2 (1 sub): {res}")
    # distance=1, total_char=5 -> CER = 0.2
    
    metric.reset()
    
    # Test case 3: Multiple samples
    preds = [('abc', 0.9), ('def', 0.8)]
    labels = [('abb', None), ('de', None)]
    res = metric((preds, labels))
    print(f"Test 3 (Mixed): {res}")
    # Sample 1: dist=1, len=3
    # Sample 2: dist=1, len=2
    # Total: dist=2, len=5 -> CER = 0.4
    
    metric.reset()

if __name__ == "__main__":
    test_cer()
