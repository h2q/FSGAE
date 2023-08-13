import random
import gmpy2
from intersect.tools import IntersectTools
class IntersectGuest(object):
    def __init__(self, rsa_public_key, random_bit=1024):
        super().__init__()
        self.rsa_public_key = rsa_public_key
        self.random_bit = random_bit
        self.random_r = None
        self.dataset = None
    def send_guest_idx(self, dataset):
        self.dataset = dataset
        n = self.rsa_public_key.n
        e = self.rsa_public_key.e
        self.random_r = list(map(lambda x: random.SystemRandom().getrandbits(self.random_bit), dataset))
        hid = list(map(lambda x: int(IntersectTools.hash(x), 16), dataset))
        guest_idx = list(map(lambda x: x[1] * gmpy2.powmod(x[0], e, n), zip(self.random_r, hid)))
        return guest_idx
    def process_host_guest_idx(self, host_idx, guest_idx_host):
        n = self.rsa_public_key.n
        guest_idx = list(
            map(lambda x: IntersectTools.hash(gmpy2.divm(x[0], x[1], n)), zip(guest_idx_host, self.random_r)))
        intersect_idx_map = dict(zip(guest_idx, self.dataset))
        intersect_idx_enc = set(host_idx) & set(guest_idx)
        intersect_idx_raw = set(map(lambda x: intersect_idx_map[x], intersect_idx_enc))
        return (intersect_idx_enc, intersect_idx_raw)
