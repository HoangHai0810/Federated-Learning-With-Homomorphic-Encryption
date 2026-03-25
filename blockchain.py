import hashlib
import json
import time
import pickle
import os
from typing import List, Any, Tuple

class Block:
    def __init__(self, index: int, previous_hash: str, data: Any,
                 timestamp: float, hash: str, nonce: int = 0):
        self.index = index
        self.previous_hash = previous_hash
        self.data = data
        self.timestamp = timestamp
        self.hash = hash
        self.nonce = nonce

class Blockchain:
    """
    Single-node, in-memory permissioned blockchain for FL model versioning.
    Uses Proof-of-Work (PoW) consensus to ensure tamper-resistance.
    Full aggregated model parameters are stored in each block's data field.
    """

    DEFAULT_DIFFICULTY = 4  # Number of leading zeros required (difficulty threshold)

    def __init__(self, difficulty: int = DEFAULT_DIFFICULTY):
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(
            index=0,
            previous_hash="0",
            data="Genesis Block",
            timestamp=time.time(),
            hash=self.calculate_hash(0, "0", "Genesis Block", time.time(), 0),
            nonce=0
        )
        self.chain.append(genesis_block)

    def _mine(self, index: int, previous_hash: str, data: Any, timestamp: float) -> Tuple[str, int]:
        """Proof-of-Work: hash data once, then iterate nonces on the small combined string."""
        prefix = '0' * self.difficulty
        # Pre-hash the large data payload ONCE outside the loop
        data_hash = hashlib.sha256(str(data).encode('utf-8')).hexdigest()
        base = f"{index}{previous_hash}{data_hash}{timestamp}"
        nonce = 0
        while True:
            candidate_hash = hashlib.sha256(f"{base}{nonce}".encode('utf-8')).hexdigest()
            if candidate_hash.startswith(prefix):
                return candidate_hash, nonce
            nonce += 1

    def add_block(self, index: int, data: Any):
        """
        Mine and append a new block with PoW consensus.
        `data` stores the full aggregated model parameters (numpy arrays) for that FL round.
        """
        last_block = self.chain[-1]
        new_timestamp = time.time()

        print(f"  [Blockchain] Mining block {index} (difficulty={self.difficulty})...")
        mine_start = time.time()
        new_hash, nonce = self._mine(index, last_block.hash, data, new_timestamp)
        mine_time = time.time() - mine_start
        print(f"  [Blockchain] Block {index} mined in {mine_time:.2f}s | nonce={nonce} | hash={new_hash[:16]}...")

        new_block = Block(
            index=index,
            previous_hash=last_block.hash,
            data=data,
            timestamp=new_timestamp,
            hash=new_hash,
            nonce=nonce
        )
        self.chain.append(new_block)

    def calculate_hash(self, index: int, previous_hash: str, data: Any,
                       timestamp: float, nonce: int = 0) -> str:
        block_string = json.dumps({
            "index": index,
            "previous_hash": previous_hash,
            "data": str(data),
            "timestamp": timestamp,
            "nonce": nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()

    def is_valid(self) -> bool:
        """Verify chain integrity: hash linkage + PoW difficulty for every block."""
        prefix = '0' * self.difficulty
        for i in range(1, len(self.chain)):
            cur = self.chain[i]
            prev = self.chain[i - 1]
            if cur.previous_hash != prev.hash:
                return False
            expected = self.calculate_hash(cur.index, cur.previous_hash,
                                           cur.data, cur.timestamp, cur.nonce)
            if cur.hash != expected:
                return False
            if not cur.hash.startswith(prefix):
                return False
        return True

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def save(self, path: str = "blockchain.pkl"):
        """Persist the chain to disk so it survives across runs."""
        with open(path, 'wb') as f:
            pickle.dump(self.chain, f)
        print(f"  [Blockchain] Chain saved to {path} ({len(self.chain)} blocks)")

    def load(self, path: str = "blockchain.pkl"):
        """Load a previously saved chain from disk."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.chain = pickle.load(f)
            print(f"  [Blockchain] Chain loaded from {path} ({len(self.chain)} blocks)")
        else:
            print(f"  [Blockchain] No saved chain found at {path}, starting fresh.")

    def print_blockchain(self):
        for block in self.chain:
            print(f"Index: {block.index} | Timestamp: {block.timestamp:.2f} | "
                  f"Nonce: {block.nonce} | Hash: {block.hash[:16]}... | "
                  f"PrevHash: {block.previous_hash[:16]}...")
