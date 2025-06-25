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

        # Time filter setup
        self.time_filters = {
            "Past 1 Month": pd.DateOffset(months=1),
            "Past 3 Months": pd.DateOffset(months=3),
            "Past 6 Months": pd.DateOffset(months=6),
            "Past 1 Year": pd.DateOffset(years=1),
            "All Time": None
        }
        self.time_filter_var = tk.StringVar(value="All Time")
        ctk.CTkLabel(self.graph_frame, text="Filter by Time Period:").pack(pady=5)
        self.time_filter_menu = ctk.CTkOptionMenu(self.graph_frame, variable=self.time_filter_var,
                                                  values=list(self.time_filters.keys()),
                                                  command=lambda x: self.update_graph())
        self.time_filter_menu.pack(pady=5)

        # Settings filter setup
        self.filter_var = tk.StringVar(value="All")
        ctk.CTkLabel(self.graph_frame, text="Filter by Settings:").pack(pady=5)
        self.filter_menu = ctk.CTkOptionMenu(self.graph_frame, variable=self.filter_var,
                                             values=["All"], command=lambda x: self.update_graph())
        self.filter_menu.pack(pady=5)

        self.update_graph()

    def update_graph(self):
        """Update the progression graph based on the time period and settings filters."""
        if not pd.io.common.file_exists("game_results.csv"):
            return

        df = pd.read_csv("game_results.csv", parse_dates=["timestamp"])

        # Filter by time period
        selected_time_period = self.time_filter_var.get()
        if selected_time_period != "All Time":
            offset = self.time_filters[selected_time_period]
            start_date = pd.Timestamp.now() - offset
            df_filtered = df[df['timestamp'] >= start_date]
        else:
            df_filtered = df

        # Get unique settings from the filtered dataframe
        unique_settings = df_filtered[["addition", "subtraction", "addition_reverse", "multiplication",
                                       "multiplication_reverse", "add_min", "add_max", "mul_min",
                                       "mul_max", "duration"]].drop_duplicates()
        self.settings_dict = {}
        for _, row in unique_settings.iterrows():
            settings_str = settings_to_str(row)
            self.settings_dict[settings_str] = row.to_dict()
        settings_list = ["All"] + list(self.settings_dict.keys())
        self.filter_menu.configure(values=settings_list)

        # Get selected settings and plot
        selected_settings = self.filter_var.get()
        plot_graph(self.ax, df_filtered, self.settings_dict, selected_settings)
        self.canvas.draw()

    def run(self):
        """Start the application."""
        self.root.mainloop()

if __name__ == "__main__":
    app = ZetamacGameApp()
    app.run()