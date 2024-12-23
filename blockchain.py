import hashlib
import json
from typing import List, Any

class Block:
    def __init__(self, index: int, previous_hash: str, data: Any, timestamp: float, hash: str):
        self.index = index
        self.previous_hash = previous_hash
        self.data = data  # Có thể chứa bất kỳ dữ liệu nào (vd: trọng số, metrics, v.v.)
        self.timestamp = timestamp
        self.hash = hash

class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(
            0, 
            "0", 
            "Genesis Block", 
            0.0, 
            self.calculate_hash(0, "0", "Genesis Block", 0.0)
        )
        self.chain.append(genesis_block)

    def add_block(self, index: int, data: Any):
        """
        Thêm block mới vào chuỗi.
        Args:
            index (int): Chỉ số vòng lặp hoặc số thứ tự block.
            data (Any): Dữ liệu cần lưu vào block (vd: trọng số đã tổng hợp).
        """
        last_block = self.chain[-1]
        new_timestamp = 1.0  # Có thể thay bằng `time.time()` nếu cần thời gian thực.
        new_hash = self.calculate_hash(index, last_block.hash, data, new_timestamp)
        new_block = Block(index, last_block.hash, data, new_timestamp, new_hash)
        self.chain.append(new_block)

    def calculate_hash(self, index: int, previous_hash: str, data: Any, timestamp: float) -> str:
        """
        Tính hash SHA-256 cho block.
        """
        block_string = json.dumps({
            "index": index,
            "previous_hash": previous_hash,
            "data": str(data), 
            "timestamp": timestamp
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def print_blockchain(self):
        """
        In ra toàn bộ blockchain.
        """
        for block in self.chain:
            print(f"Index: {block.index}, Timestamp: {block.timestamp}, Hash: {block.hash}, Previous Hash: {block.previous_hash}, Data: {block.data}")
