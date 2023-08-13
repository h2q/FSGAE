from intersect.guest import IntersectGuest
from intersect.host import IntersectHost
class Intersect(object):
    def run(self, dataset_host, dataset_guest):
        host = IntersectHost()
        guest = IntersectGuest(host.get_rsa_public_key())
        guest_idx = guest.send_guest_idx(dataset_guest)
        guest_idx_host = host.process_guest_idx(guest_idx)
        host_idx = host.send_host_idx(dataset_host)
        (intersect_idx_enc, intersect_idx_raw_guest) = guest.process_host_guest_idx(host_idx, guest_idx_host)
        intersect_idx_raw_host = host.process_intersect_idx(intersect_idx_enc)
        assert (intersect_idx_raw_host == intersect_idx_raw_guest)
        return intersect_idx_raw_host
