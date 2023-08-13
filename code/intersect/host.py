import rsa
import gmpy2
from intersect.tools import IntersectTools
class IntersectHost(object):
    def __init__(self, rsa_key_bits=1024):
        self.rsa_public_key, self.rsa_secret_key = rsa.newkeys(rsa_key_bits)
        self.host_idx_map = None
    def get_rsa_public_key(self):
        return self.rsa_public_key
    def process_guest_idx(self, guest_idx):
        n = self.rsa_public_key.n
        d = self.rsa_secret_key.d
        guest_idx_host = list(map(lambda x: gmpy2.powmod(int(x), d, n), guest_idx))
        return guest_idx_host
    def send_host_idx(self, dataset):
        n = self.rsa_public_key.n
        d = self.rsa_secret_key.d
        host_idx = list(
            map(lambda x: IntersectTools.hash(gmpy2.powmod(int(IntersectTools.hash(x), 16), d, n)), dataset))
        self.host_idx_map = dict(zip(host_idx, dataset))
        return host_idx
    def process_intersect_idx(self, intersect_idx_enc):
        intersect_idx_raw = set(map(lambda x: self.host_idx_map[x], intersect_idx_enc))
        return intersect_idx_raw
