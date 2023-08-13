import hashlib
class IntersectTools(object):
    @staticmethod
    def hash(value):
        return hashlib.sha3_256(bytes(str(value), encoding='utf-8')).hexdigest()
