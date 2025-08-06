class TranspositionTable:
    def __init__(self):
        self.table = {}

    def get(self, zobrist_hash):
        return self.table.get(zobrist_hash, None)

    def put(self, zobrist_hash, value):
        self.table[zobrist_hash] = value
