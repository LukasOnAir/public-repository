# settings.py
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from game import GameFrame

class SettingsFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # Problem types and ranges
        self.problem_types = {
            "Addition": [tk.BooleanVar(value=True), ctk.CTkEntry(self, width=50), ctk.CTkEntry(self, width=50)],
            "Subtraction": [tk.BooleanVar(value=True)],
            "Addition problems in reverse": [tk.BooleanVar(value=True)],
            "Multiplication": [tk.BooleanVar(value=True), ctk.CTkEntry(self, width=50), ctk.CTkEntry(self, width=50)],
            "Division": [tk.BooleanVar(value=True)]
        }

        row = 0
        for name, vars in self.problem_types.items():
            checkbox = ctk.CTkCheckBox(self, text=name, variable=vars[0])
            checkbox.grid(row=row, column=0, padx=5, pady=5, sticky="w")
            if len(vars) > 1:  # Range entries for Addition and Multiplication
                min_entry, max_entry = vars[1], vars[2]
                min_entry.insert(0, "2")
                max_entry.insert(0, "100" if name == "Addition" else "12")
                ctk.CTkLabel(self, text="Range (").grid(row=row, column=1)
                min_entry.grid(row=row, column=2)
                ctk.CTkLabel(self, text=" to ").grid(row=row, column=3)
                max_entry.grid(row=row, column=4)
                ctk.CTkLabel(self, text=")").grid(row=row, column=5)
            row += 1

        # Duration dropdown
        ctk.CTkLabel(self, text="Duration:").grid(row=row, column=0, pady=5, sticky="w")
        self.duration_options = ["30 seconds", "60 seconds", "120 seconds", "300 seconds", "600 seconds"]
        self.duration_var = tk.StringVar(value="120 seconds")
        duration_menu = ctk.CTkOptionMenu(self, values=self.duration_options, variable=self.duration_var)
        duration_menu.grid(row=row, column=1, columnspan=2, pady=5)

        # Start button
        self.start_button = ctk.CTkButton(self, text="Start", command=self.start_game)
        self.start_button.grid(row=row + 1, column=0, columnspan=6, pady=10)

    def get_settings(self):
        """Retrieve the current settings."""
        settings = {
            "addition": self.problem_types["Addition"][0].get(),
            "subtraction": self.problem_types["Subtraction"][0].get(),
            "addition_reverse": self.problem_types["Addition problems in reverse"][0].get(),
            "multiplication": self.problem_types["Multiplication"][0].get(),
            "multiplication_reverse": self.problem_types["Division"][0].get(),
            "add_min": int(self.problem_types["Addition"][1].get()),
            "add_max": int(self.problem_types["Addition"][2].get()),
            "mul_min": int(self.problem_types["Multiplication"][1].get()),
            "mul_max": int(self.problem_types["Multiplication"][2].get()),
            "duration": int(self.duration_var.get().split()[0])
        }
        return settings

    def start_game(self):
        """Validate settings and start the game."""
        try:
            settings = self.get_settings()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for ranges.")
            return

        if not any([settings["addition"], settings["subtraction"], settings["addition_reverse"],
                    settings["multiplication"], settings["multiplication_reverse"]]):
            messagebox.showerror("Error", "Select at least one problem type.")
            return
        if settings["add_min"] >= settings["add_max"] or settings["mul_min"] >= settings["mul_max"]:
            messagebox.showerror("Error", "Minimum must be less than maximum for ranges.")
            return

        self.pack_forget()
        self.app.graph_frame.pack_forget()  # Hide the graph frame when the game starts
        self.app.game_frame = GameFrame(self.app.left_frame, self.app, settings)
        self.app.game_frame.pack(fill="both", expand=True)