from phe import paillier

class PaillierEncryption:
    def __init__(self):
        self.public_key, self.private_key = paillier.generate_paillier_keypair()

    def encrypt(self, param):
        if isinstance(param, list):
            return [self.public_key.encrypt(p) for p in param]
        return self.public_key.encrypt(param)

    def decrypt(self, encrypted_param):
        return self.private_key.decrypt(encrypted_param)