#main
import customtkinter as ctk
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from settings import *
from utils import settings_to_str, plot_graph
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from settings import SettingsFrame

class ZetamacGameApp:
    def __init__(self):
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Arithmetic Game")
        self.root.geometry("800x600")

        # Title and description
        ctk.CTkLabel(self.root, text="Arithmetic Game", font=("Arial", 20, "bold")).pack(pady=10)
        ctk.CTkLabel(self.root, text="Solve as many arithmetic problems as you can within the time limit!",
                     font=("Arial", 12)).pack(pady=5)

        # Layout frames
        self.left_frame = ctk.CTkFrame(self.root)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=10)

        self.graph_frame = ctk.CTkFrame(self.root)
        self.graph_frame.pack(side="right", fill="both", expand=True, padx=10)

        # Settings frame
        self.settings_frame = SettingsFrame(self.left_frame, self)
        self.settings_frame.pack(fill="both", expand=True)

        # Graph setup
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.filter_var = tk.StringVar(value="All")
        ctk.CTkLabel(self.graph_frame, text="Filter by Settings:").pack(pady=5)
        self.filter_menu = ctk.CTkOptionMenu(self.graph_frame, variable=self.filter_var,
                                             values=["All"], command=lambda x: self.update_graph())
        self.filter_menu.pack(pady=5)

        self.update_graph()

    def update_graph(self):
        """Update the progression graph based on the filter."""
        if not pd.io.common.file_exists("game_results.csv"):
            return

        df = pd.read_csv("game_results.csv", parse_dates=["timestamp"])
        unique_settings = df[["addition", "subtraction", "addition_reverse", "multiplication",
                              "multiplication_reverse", "add_min", "add_max", "mul_min",
                              "mul_max", "duration"]].drop_duplicates()

        self.settings_dict = {}
        for _, row in unique_settings.iterrows():
            settings_str = settings_to_str(row)
            self.settings_dict[settings_str] = row.to_dict()

        settings_list = ["All"] + list(self.settings_dict.keys())
        self.filter_menu.configure(values=settings_list)

        plot_graph(self.ax, df, self.settings_dict, self.filter_var.get())
        self.canvas.draw()

    def run(self):
        """Start the application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = ZetamacGameApp()
    app.run()