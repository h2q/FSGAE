import unittest
import numpy as np
from intersect.driver import Intersect
import time
class TestIntersect(unittest.TestCase):
    def test_intersect(self):
        begin_time = time.clock()
        dataset_guest = np.array([1, 2, 3, 4, 5])
        dataset_host = np.array([2, 3, 5, 7, 9])
        dataset_guest = np.append(dataset_guest, np.random.randint(10, int(1e6), int(1e3)))
        dataset_host = np.append(dataset_host, np.random.randint(10, int(1e6), int(1e3)))
        intersect = Intersect()
        intersect_idx_raw = intersect.run(dataset_host, dataset_guest)
        end_time = time.clock()
        
        
if __name__ == '__main__':
    unittest.main()
