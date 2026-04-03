from agents.base_agent import BaseAgent

class MiniMaxAgent(BaseAgent):
    def __init__(self, player: int, name: str = "MiniMax", max_depth: int = None):
        super().__init__(player, name)
        self.max_depth = max_depth
        self.nodes_explored = 0

    def get_move(self, game) -> int:
        self.nodes_explored = 0
        valid_moves = game.get_valid_moves()
        best_move = valid_moves[0]
        best_score = float('-inf')

        for move in valid_moves:
            g = game.clone()
            g.make_move(move, self.player)
            score = self._minimax(g, 1, False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _minimax(self, game, depth, is_maximizing) -> float:
        self.nodes_explored += 1
        winner = game.check_winner()
        if winner is not None:
            if winner == self.player:
                return 10000 - depth
            elif winner == -self.player:
                return depth - 10000
            else:
                return 0

        if self.max_depth is not None and depth >= self.max_depth:
            return self._evaluate(game)

        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return 0

        current_player = self.player if is_maximizing else -self.player

        if is_maximizing:
            best = float('-inf')
            for move in valid_moves:
                g = game.clone()
                g.make_move(move, current_player)
                score = self._minimax(g, depth + 1, False)
                best = max(best, score)
            return best
        else:
            best = float('inf')
            for move in valid_moves:
                g = game.clone()
                g.make_move(move, current_player)
                score = self._minimax(g, depth + 1, True)
                best = min(best, score)
            return best

    def _evaluate(self, game) -> float:
        from games.connect4 import Connect4
        if not isinstance(game, Connect4):
            return 0
        return self._connect4_heuristic(game)

    def _connect4_heuristic(self, game) -> float:
        score = 0
        b = game.board
        rows, cols = b.shape

        def score_window(window, player):
            opp = -player
            s = 0
            pc = list(window).count(player)
            ec = list(window).count(0)
            oc = list(window).count(opp)
            if pc == 4:
                s += 10000
            elif pc == 3 and ec == 1:
                s += 100
            elif pc == 2 and ec == 2:
                s += 10
            if oc == 3 and ec == 1:
                s -= 80
            return s

                                  
        center_col = list(b[:, cols // 2])
        score += center_col.count(self.player) * 5
        score -= center_col.count(-self.player) * 5

                    
        for r in range(rows):
            for c in range(cols - 3):
                window = b[r, c:c+4]
                score += score_window(window, self.player)

                  
        for r in range(rows - 3):
            for c in range(cols):
                window = b[r:r+4, c]
                score += score_window(window, self.player)

                    
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = [b[r+i, c+i] for i in range(4)]
                score += score_window(window, self.player)

                    
        for r in range(rows - 3):
            for c in range(3, cols):
                window = [b[r+i, c-i] for i in range(4)]
                score += score_window(window, self.player)

        return score
