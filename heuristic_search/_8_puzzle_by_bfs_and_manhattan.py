import tkinter as tk
from tkinter import messagebox
import random
from collections import deque
from copy import deepcopy

GOAL = [[1,2,3],[4,5,6],[7,8,0]]
CELL_SIZE = 100

def manhattan(state):
    dist = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                goal_x, goal_y = divmod(val - 1, 3)
                dist += abs(goal_x - i) + abs(goal_y - j)
    return dist

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def move(state, direction):
    x, y = find_blank(state)
    dx, dy = {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}[direction]
    nx, ny = x+dx, y+dy
    if 0 <= nx < 3 and 0 <= ny < 3:
        new = deepcopy(state)
        new[x][y], new[nx][ny] = new[nx][ny], new[x][y]
        return new
    return None

def serialize(state):
    return tuple(tuple(row) for row in state)

def is_solvable(puzzle):
    inv = 0
    arr = [x for x in puzzle if x != 0]
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv % 2 == 0

def generate_random_state():
    flat = list(range(9))
    while True:
        random.shuffle(flat)
        state = [flat[i:i+3] for i in range(0,9,3)]
        if is_solvable(flat):
            return state

def solve_bfs(start):
    visited = set()
    queue = deque()
    queue.append((start, [], manhattan(start)))
    visited.add(serialize(start))
    while queue:
        state, path, h = queue.popleft()
        if state == GOAL:
            return path
        for dir in ['up','down','left','right']:
            new_state = move(state, dir)
            if new_state and serialize(new_state) not in visited:
                visited.add(serialize(new_state))
                queue.append((new_state, path + [(dir, new_state)], manhattan(new_state)))
    return []

class PuzzleApp:
    def __init__(self, master):
        self.master = master
        self.master.title("8 Puzzle - Step by Step")
        self.canvas = tk.Canvas(master, width=3*CELL_SIZE, height=3*CELL_SIZE, bg="white")
        self.canvas.pack(pady=10)

        self.next_button = tk.Button(master, text="Next Move ▶️", command=self.next_step)
        self.next_button.pack(pady=5)

        self.restart_button = tk.Button(master, text="Shuffle Again 🔄", command=self.restart)
        self.restart_button.pack(pady=5)

        self.state = []
        self.steps = []
        self.index = 0
        self.restart()

    def draw_grid(self, move_text="Start"):
        self.canvas.delete("all")
        for i in range(3):
            for j in range(3):
                val = self.state[i][j]
                x0, y0 = j * CELL_SIZE, i * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                color = "lightblue" if val != 0 else "white"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
                if val != 0:
                    self.canvas.create_text((x0+x1)//2, (y0+y1)//2, text=str(val), font=("Arial", 24), fill="black")
        self.master.title(f"8 Puzzle - {move_text} | h: {manhattan(self.state)}")

    def restart(self):
        self.state = generate_random_state()
        self.steps = solve_bfs(self.state)
        self.index = 0
        self.draw_grid("Start")

    def next_step(self):
        if self.index < len(self.steps):
            move_name, new_state = self.steps[self.index]
            self.state = new_state
            self.draw_grid(move_name)
            self.index += 1
        else:
            messagebox.showinfo("Done", "🎉 Puzzle Solved!")

# --- Launch App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()