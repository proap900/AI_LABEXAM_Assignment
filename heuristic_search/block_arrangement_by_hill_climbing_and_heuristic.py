import tkinter as tk
from tkinter import messagebox
import random

# --- Constants ---
GOAL = ['A', 'B', 'C', 'D']
COLORS = {'A': 'red', 'B': 'green', 'C': 'blue', 'D': 'orange'}
BLOCK_SIZE = 60
BLOCK_GAP = 10

class BlockArrangementApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Block Arrangement - Hill Climbing")
        self.canvas = tk.Canvas(master, width=200, height=300, bg="white")
        self.canvas.pack(pady=10)

        self.next_button = tk.Button(master, text="Next Swap 🔄", command=self.next_step)
        self.next_button.pack(pady=5)

        self.restart_button = tk.Button(master, text="Shuffle Again 🎲", command=self.restart)
        self.restart_button.pack(pady=5)

        self.state = []
        self.h = 0
        self.done = False
        self.restart()

    def heuristic(self, state):
        return sum(1 for i in range(4) if state[i] != GOAL[i])

    def neighbors(self, state):
        result = []
        for i in range(len(state) - 1):
            new = state[:]
            new[i], new[i+1] = new[i+1], new[i]
            result.append((new, f"Swap {i}-{i+1}"))
        return result

    def draw_stack(self, move=None):
        self.canvas.delete("all")
        for i, block in enumerate(self.state):
            x0 = 70
            y0 = 30 + i * (BLOCK_SIZE + BLOCK_GAP)
            x1 = x0 + BLOCK_SIZE
            y1 = y0 + BLOCK_SIZE
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=COLORS[block], outline="black")
            self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=block, fill="white", font=("Arial", 16))

        title = f"{move} | Heuristic: {self.h}" if move else f"Start | Heuristic: {self.h}"
        self.master.title(title)

    def restart(self):
        self.state = self.random_stack()
        self.h = self.heuristic(self.state)
        self.done = False
        self.draw_stack("Start")

    def random_stack(self):
        st = GOAL[:]
        while True:
            random.shuffle(st)
            if st != GOAL:
                return st

    def next_step(self):
        if self.done:
            return
        for neighbor, move in self.neighbors(self.state):
            h_new = self.heuristic(neighbor)
            if h_new < self.h:
                self.state = neighbor
                self.h = h_new
                self.draw_stack(move)
                if self.h == 0:
                    self.done = True
                    messagebox.showinfo("Success", "🎉 Goal Reached!")
                return
        self.done = True
        messagebox.showwarning("Stuck", "⚠️ Stuck in local maximum!\nClick 'Shuffle Again' to restart.")

# --- Launch ---
if __name__ == "__main__":
    root = tk.Tk()
    app = BlockArrangementApp(root)
    root.mainloop()