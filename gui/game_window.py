"""
GUI windows for Tic Tac Toe and Connect 4 using tkinter.

Usage:
    TicTacToeWindow(agent1, agent2).run()
    Connect4Window(agent1, agent2).run()

For human play, pass HumanGUIAgent as agent1 or agent2.
"""

import tkinter as tk


# ---------------------------------------------------------------------------
# Human placeholder agent (moves are injected by click events)
# ---------------------------------------------------------------------------

class HumanGUIAgent:
    """Signals that a human is playing via the GUI. Clicks are handled by the window."""
    def __init__(self, player: int, name: str = "Human"):
        self.player = player
        self.name = name

    def get_move(self, game) -> int:
        pass  # handled by GUI


# ---------------------------------------------------------------------------
# Tic Tac Toe Window
# ---------------------------------------------------------------------------

class TicTacToeWindow:
    CELL = 140
    PAD = 30
    BG = "#1a1a2e"
    BOARD_BG = "#16213e"
    LINE_COLOR = "#e94560"
    X_COLOR = "#00b4d8"
    O_COLOR = "#f5a623"
    TEXT_COLOR = "#eaeaea"

    def __init__(self, agent1, agent2, move_delay: int = 700):
        """
        agent1 = player 1 (X), agent2 = player -1 (O).
        move_delay: ms pause between AI moves (ignored for human moves).
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.move_delay = move_delay
        self._human_move = None
        self._waiting_for_human = False

    def run(self):
        from games.tic_tac_toe import TicTacToe
        self.game = TicTacToe()
        self.game.reset()

        board_w = self.CELL * 3 + self.PAD * 2
        board_h = self.CELL * 3 + self.PAD * 2

        self.root = tk.Tk()
        self.root.title("Tic Tac Toe — AI Game Agents")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        # Title label
        tk.Label(
            self.root,
            text=f"{self.agent1.name}  (X)   vs   {self.agent2.name}  (O)",
            font=("Helvetica", 14, "bold"),
            bg=self.BG, fg=self.TEXT_COLOR,
        ).pack(pady=(14, 4))

        # Status label
        self.status_var = tk.StringVar(value="Game starting…")
        tk.Label(
            self.root, textvariable=self.status_var,
            font=("Helvetica", 12),
            bg=self.BG, fg="#aaaaaa",
        ).pack(pady=(0, 8))

        # Canvas
        self.canvas = tk.Canvas(
            self.root, width=board_w, height=board_h,
            bg=self.BOARD_BG, highlightthickness=2,
            highlightbackground=self.LINE_COLOR,
        )
        self.canvas.pack(padx=24, pady=(0, 20))
        self.canvas.bind("<Button-1>", self._on_click)

        self._draw_grid()
        self.root.after(400, self._next_move)
        self.root.mainloop()

    # --- Drawing helpers ---

    def _draw_grid(self):
        p, c = self.PAD, self.CELL
        for i in range(1, 3):
            x = p + i * c
            self.canvas.create_line(x, p, x, p + 3 * c, fill=self.LINE_COLOR, width=3)
            y = p + i * c
            self.canvas.create_line(p, y, p + 3 * c, y, fill=self.LINE_COLOR, width=3)

        # cell hover highlight rectangles (hidden initially)
        for pos in range(9):
            row, col = divmod(pos, 3)
            x0 = p + col * c + 4
            y0 = p + row * c + 4
            x1 = x0 + c - 8
            y1 = y0 + c - 8
            self.canvas.create_rectangle(
                x0, y0, x1, y1,
                outline="", fill="",
                tags=f"hover_{pos}",
            )

    def _draw_piece(self, pos: int, player: int):
        p, c = self.PAD, self.CELL
        row, col = divmod(pos, 3)
        cx = p + col * c + c // 2
        cy = p + row * c + c // 2
        r = c // 2 - 18

        if player == 1:  # X
            self.canvas.create_line(
                cx - r, cy - r, cx + r, cy + r,
                fill=self.X_COLOR, width=8, capstyle="round",
            )
            self.canvas.create_line(
                cx + r, cy - r, cx - r, cy + r,
                fill=self.X_COLOR, width=8, capstyle="round",
            )
        else:  # O
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=self.O_COLOR, width=8,
            )

    def _highlight_winner(self):
        """Draw a line through the winning three cells."""
        b = self.game.board.reshape(3, 3)
        p, c = self.PAD, self.CELL

        def cell_center(r, col):
            return p + col * c + c // 2, p + r * c + c // 2

        lines_to_check = (
            [(0, i) for i in range(3)],
            [(1, i) for i in range(3)],
            [(2, i) for i in range(3)],
            [(i, 0) for i in range(3)],
            [(i, 1) for i in range(3)],
            [(i, 2) for i in range(3)],
            [(i, i) for i in range(3)],
            [(i, 2 - i) for i in range(3)],
        )
        for line in lines_to_check:
            vals = [b[r][col] for r, col in line]
            if abs(sum(vals)) == 3 and 0 not in vals:
                x0, y0 = cell_center(*line[0])
                x1, y1 = cell_center(*line[2])
                self.canvas.create_line(
                    x0, y0, x1, y1,
                    fill="white", width=6, capstyle="round",
                )
                break

    # --- Game loop ---

    def _current_agent(self):
        return self.agent1 if self.game.current_player == 1 else self.agent2

    def _next_move(self):
        agent = self._current_agent()
        if isinstance(agent, HumanGUIAgent):
            self.status_var.set("Your turn — click a cell")
            self._waiting_for_human = True
        else:
            self.status_var.set(f"{agent.name} is thinking…")
            self.root.after(self.move_delay, self._make_ai_move)

    def _make_ai_move(self):
        agent = self._current_agent()
        move = agent.get_move(self.game)
        self._apply_move(move)

    def _on_click(self, event):
        if not self._waiting_for_human:
            return
        p, c = self.PAD, self.CELL
        col = (event.x - p) // c
        row = (event.y - p) // c
        if 0 <= row < 3 and 0 <= col < 3:
            move = row * 3 + col
            if move in self.game.get_valid_moves():
                self._waiting_for_human = False
                self._apply_move(move)

    def _apply_move(self, move: int):
        player = self.game.current_player
        _, _, done, info = self.game.make_move(move, player)
        self._draw_piece(move, player)

        if done:
            winner = info.get("winner")
            if winner == 1:
                msg = f"🎉  {self.agent1.name} (X) wins!"
                self._highlight_winner()
            elif winner == -1:
                msg = f"🎉  {self.agent2.name} (O) wins!"
                self._highlight_winner()
            else:
                msg = "It's a draw!"
            self.status_var.set(msg)
            self.root.after(2500, self.root.destroy)
        else:
            self.root.after(120, self._next_move)


# ---------------------------------------------------------------------------
# Connect 4 Window
# ---------------------------------------------------------------------------

class Connect4Window:
    CELL = 82
    PAD = 24
    ROWS = 6
    COLS = 7
    BG = "#1a1a2e"
    BOARD_BG = "#0f3460"
    EMPTY_COLOR = "#16213e"
    P1_COLOR = "#e94560"   # red
    P2_COLOR = "#f5a623"   # yellow
    TEXT_COLOR = "#eaeaea"

    def __init__(self, agent1, agent2, move_delay: int = 700):
        self.agent1 = agent1
        self.agent2 = agent2
        self.move_delay = move_delay
        self._human_move = None
        self._waiting_for_human = False
        self._hover_col = -1

    def run(self):
        from games.connect4 import Connect4
        self.game = Connect4()
        self.game.reset()

        board_w = self.CELL * self.COLS + self.PAD * 2
        board_h = self.CELL * self.ROWS + self.PAD * 2

        self.root = tk.Tk()
        self.root.title("Connect 4 — AI Game Agents")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        tk.Label(
            self.root,
            text=f"{self.agent1.name}  (●Red)   vs   {self.agent2.name}  (●Yellow)",
            font=("Helvetica", 13, "bold"),
            bg=self.BG, fg=self.TEXT_COLOR,
        ).pack(pady=(14, 4))

        self.status_var = tk.StringVar(value="Game starting…")
        tk.Label(
            self.root, textvariable=self.status_var,
            font=("Helvetica", 12),
            bg=self.BG, fg="#aaaaaa",
        ).pack(pady=(0, 6))

        # Arrow row for hover indication
        self.arrow_canvas = tk.Canvas(
            self.root, width=board_w, height=28,
            bg=self.BG, highlightthickness=0,
        )
        self.arrow_canvas.pack()

        self.canvas = tk.Canvas(
            self.root, width=board_w, height=board_h,
            bg=self.BOARD_BG, highlightthickness=2,
            highlightbackground="#e94560",
        )
        self.canvas.pack(padx=24, pady=(2, 20))
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Leave>", self._on_leave)

        self._draw_empty_board()
        self.root.after(400, self._next_move)
        self.root.mainloop()

    # --- Drawing helpers ---

    def _cell_center(self, row: int, col: int):
        p, c = self.PAD, self.CELL
        return p + col * c + c // 2, p + row * c + c // 2

    def _draw_empty_board(self):
        r = self.CELL // 2 - 6
        for row in range(self.ROWS):
            for col in range(self.COLS):
                cx, cy = self._cell_center(row, col)
                self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    fill=self.EMPTY_COLOR, outline=self.BOARD_BG, width=3,
                    tags=f"cell_{row}_{col}",
                )

    def _draw_piece(self, row: int, col: int, player: int):
        r = self.CELL // 2 - 6
        cx, cy = self._cell_center(row, col)
        color = self.P1_COLOR if player == 1 else self.P2_COLOR
        self.canvas.delete(f"cell_{row}_{col}")
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=color, outline="white", width=2,
            tags=f"cell_{row}_{col}",
        )

    def _draw_hover_arrow(self, col: int):
        self.arrow_canvas.delete("arrow")
        if col < 0 or col >= self.COLS:
            return
        cx, _ = self._cell_center(0, col)
        player = self.game.current_player
        color = self.P1_COLOR if player == 1 else self.P2_COLOR
        # downward triangle
        self.arrow_canvas.create_polygon(
            cx, 24, cx - 12, 6, cx + 12, 6,
            fill=color, tags="arrow",
        )

    def _highlight_winning_cells(self):
        b = self.game.board

        def check_line(cells):
            vals = [b[r][c] for r, c in cells]
            if abs(sum(vals)) == 4 and 0 not in vals:
                return vals[0]
            return None

        winning_cells = []
        for row in range(self.ROWS):
            for col in range(self.COLS - 3):
                cells = [(row, col + i) for i in range(4)]
                if check_line(cells):
                    winning_cells = cells
                    break
        if not winning_cells:
            for col in range(self.COLS):
                for row in range(self.ROWS - 3):
                    cells = [(row + i, col) for i in range(4)]
                    if check_line(cells):
                        winning_cells = cells
                        break
        if not winning_cells:
            for row in range(self.ROWS - 3):
                for col in range(self.COLS - 3):
                    cells = [(row + i, col + i) for i in range(4)]
                    if check_line(cells):
                        winning_cells = cells
                        break
        if not winning_cells:
            for row in range(self.ROWS - 3):
                for col in range(3, self.COLS):
                    cells = [(row + i, col - i) for i in range(4)]
                    if check_line(cells):
                        winning_cells = cells
                        break

        for row, col in winning_cells:
            r = self.CELL // 2 - 6
            cx, cy = self._cell_center(row, col)
            self.canvas.create_oval(
                cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4,
                outline="white", width=4,
            )

    # --- Game loop ---

    def _current_agent(self):
        return self.agent1 if self.game.current_player == 1 else self.agent2

    def _next_move(self):
        agent = self._current_agent()
        if isinstance(agent, HumanGUIAgent):
            color_name = "Red" if agent.player == 1 else "Yellow"
            self.status_var.set(f"Your turn ({color_name}) — click a column")
            self._waiting_for_human = True
        else:
            self.status_var.set(f"{agent.name} is thinking…")
            self.root.after(self.move_delay, self._make_ai_move)

    def _make_ai_move(self):
        agent = self._current_agent()
        move = agent.get_move(self.game)
        self._apply_move(move)

    def _on_click(self, event):
        if not self._waiting_for_human:
            return
        p, c = self.PAD, self.CELL
        col = (event.x - p) // c
        if 0 <= col < self.COLS and col in self.game.get_valid_moves():
            self._waiting_for_human = False
            self.arrow_canvas.delete("arrow")
            self._apply_move(col)

    def _on_hover(self, event):
        if not self._waiting_for_human:
            return
        p, c = self.PAD, self.CELL
        col = (event.x - p) // c
        if col != self._hover_col:
            self._hover_col = col
            self._draw_hover_arrow(col if 0 <= col < self.COLS else -1)

    def _on_leave(self, _event):
        self._hover_col = -1
        self.arrow_canvas.delete("arrow")

    def _apply_move(self, col: int):
        player = self.game.current_player
        board_before = self.game.board.copy()
        _, _, done, info = self.game.make_move(col, player)

        # find the row the piece landed in
        for row in range(self.ROWS):
            if self.game.board[row, col] != board_before[row, col]:
                self._draw_piece(row, col, player)
                break

        if done:
            winner = info.get("winner")
            if winner == 1:
                msg = f"🎉  {self.agent1.name} wins!"
                self._highlight_winning_cells()
            elif winner == -1:
                msg = f"🎉  {self.agent2.name} wins!"
                self._highlight_winning_cells()
            else:
                msg = "It's a draw!"
            self.status_var.set(msg)
            self.root.after(3000, self.root.destroy)
        else:
            self.root.after(120, self._next_move)
