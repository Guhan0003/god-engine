class UCIInterface:
    def __init__(self, board, search):
        self.board = board
        self.search = search

    def loop(self):
        while True:
            cmd = input()
            if cmd == "uci":
                print("id name GodEngine")
                print("uciok")
            elif cmd == "isready":
                print("readyok")
            elif cmd.startswith("position"):
                # TODO: parse position and update board
                pass
            elif cmd.startswith("go"):
                best_move = self.search.search(self.board, depth=5)
                print(f"bestmove {best_move}")
            elif cmd == "quit":
                break
