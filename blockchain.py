class Blockchain:
    def __init__(self):
        self.chain = []

    def add_transaction(self, transaction_hash):
        block = {"hash": transaction_hash, "index": len(self.chain)}
        self.chain.append(block)

    def validate_chain(self):
        for i in range(len(self.chain)):
            if self.chain[i]["index"] != i:
                return False
        return True
