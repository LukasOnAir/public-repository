#game
import customtkinter as ctk
import random
from tkinter import messagebox
from utils import save_game_results

class GameFrame(ctk.CTkFrame):
    def __init__(self, parent, app, settings):
        super().__init__(parent)
        self.app = app
        self.settings = settings
        self.score = 0
        self.time_left = settings["duration"]

        # UI elements
        self.problem_label = ctk.CTkLabel(self, text="", font=("Arial", 16))
        self.problem_label.pack(pady=10)

        self.answer_entry = ctk.CTkEntry(self, width=100)
        self.answer_entry.pack(pady=5)
        self.answer_entry.bind("<Return>", lambda event: self.check_answer())

        self.submit_button = ctk.CTkButton(self, text="Submit", command=self.check_answer)
        self.submit_button.pack(pady=5)

        self.timer_label = ctk.CTkLabel(self, text=f"Time Left: {self.time_left}s", font=("Arial", 12))
        self.timer_label.pack(pady=5)

        self.score_label = ctk.CTkLabel(self, text=f"Score: {self.score}", font=("Arial", 12))
        self.score_label.pack(pady=5)

        # Start game
        self.generate_problem()
        self.update_timer()

    def generate_problem(self):
        """Generate a new problem based on settings."""
        types = []
        if self.settings["addition"]: types.append("addition")
        if self.settings["subtraction"]: types.append("subtraction")
        if self.settings["addition_reverse"]: types.append("addition_reverse")
        if self.settings["multiplication"]: types.append("multiplication")
        if self.settings["multiplication_reverse"]: types.append("multiplication_reverse")

        if not types:
            return

        problem_type = random.choice(types)
        if problem_type == "addition":
            a = random.randint(self.settings["add_min"], self.settings["add_max"])
            b = random.randint(self.settings["add_min"], self.settings["add_max"])
            self.problem = f"{a} + {b} = ?"
            self.answer = a + b
        elif problem_type == "subtraction":
            a = random.randint(2, 100)
            b = random.randint(2, 100)
            a, b = max(a, b), min(a, b)
            self.problem = f"{a} - {b} = ?"
            self.answer = a - b
        elif problem_type == "addition_reverse":
            c = random.randint(self.settings["add_min"], self.settings["add_max"] * 2)
            a = random.randint(self.settings["add_min"], c)
            self.problem = f"{c} - {a} = ?"
            self.answer = c - a
        elif problem_type == "multiplication":
            a = random.randint(self.settings["mul_min"], self.settings["mul_max"])
            b = random.randint(2, 100)
            self.problem = f"{a} × {b} = ?"
            self.answer = a * b
        elif problem_type == "multiplication_reverse":
            a = random.randint(self.settings["mul_min"], self.settings["mul_max"])
            c = a * random.randint(2, 100)
            self.problem = f"{c} ÷ {a} = ?"
            self.answer = c // a

        self.problem_label.configure(text=self.problem)
        self.answer_entry.delete(0, "end")
        self.answer_entry.focus()

    def check_answer(self):
        """Check the user's answer and update score."""
        try:
            user_answer = int(self.answer_entry.get())
            if user_answer == self.answer:
                self.score += 1
                self.score_label.configure(text=f"Score: {self.score}")
            self.generate_problem()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")

    def update_timer(self):
        """Update the countdown timer."""
        if self.time_left > 0:
            self.timer_label.configure(text=f"Time Left: {self.time_left}s")
            self.time_left -= 1
            self.app.root.after(1000, self.update_timer)
        else:
            self.end_game()

    def end_game(self):
        """End the game and save results."""
        save_game_results(self.settings, self.score)
        self.pack_forget()
        self.app.settings_frame.pack(fill="both", expand=True)
        self.app.update_graph()
        messagebox.showinfo("Game Over", f"Your score: {self.score}")