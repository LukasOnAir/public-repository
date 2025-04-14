#utils
import csv
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def save_game_results(settings, score, filename="game_results.csv"):
    """Save game results to a CSV file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "addition": int(settings["addition"]),
        "subtraction": int(settings["subtraction"]),
        "addition_reverse": int(settings["addition_reverse"]),
        "multiplication": int(settings["multiplication"]),
        "multiplication_reverse": int(settings["multiplication_reverse"]),
        "add_min": settings["add_min"],
        "add_max": settings["add_max"],
        "mul_min": settings["mul_min"],
        "mul_max": settings["mul_max"],
        "duration": settings["duration"],
        "score": score
    }
    file_exists = os.path.exists(filename) and os.stat(filename).st_size > 0
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def settings_to_str(row):
    """Convert settings from a DataFrame row to a string for filtering."""
    ops = []
    if row["addition"]: ops.append("Add")
    if row["subtraction"]: ops.append("Sub")
    if row["addition_reverse"]: ops.append("AddRev")
    if row["multiplication"]: ops.append("Mul")
    if row["multiplication_reverse"]: ops.append("MulRev")
    ops_str = ",".join(ops) if ops else "None"
    ranges = f"Add:{row['add_min']}-{row['add_max']},Mul:{row['mul_min']}-{row['mul_max']}"
    return f"{ops_str} [{ranges}] {row['duration']}s"

def plot_graph(ax, df, settings_dict, selected):
    """Plot the progression graph based on the selected filter."""
    if selected == "All":
        filtered_df = df
    else:
        settings = settings_dict[selected]
        filtered_df = df[
            (df["addition"] == settings["addition"]) &
            (df["subtraction"] == settings["subtraction"]) &
            (df["addition_reverse"] == settings["addition_reverse"]) &
            (df["multiplication"] == settings["multiplication"]) &
            (df["multiplication_reverse"] == settings["multiplication_reverse"]) &
            (df["add_min"] == settings["add_min"]) &
            (df["add_max"] == settings["add_max"]) &
            (df["mul_min"] == settings["mul_min"]) &
            (df["mul_max"] == settings["mul_max"]) &
            (df["duration"] == settings["duration"])
        ]

    filtered_df = filtered_df.sort_values("timestamp")
    ax.clear()
    ax.plot(filtered_df["timestamp"], filtered_df["score"], marker="o", linestyle="-")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.set_title("Progression Graph")
    plt.xticks(rotation=45)
    plt.tight_layout()