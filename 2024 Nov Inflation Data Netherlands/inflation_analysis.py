import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Read the CSV file
df = pd.read_csv(r"Inflation Data Netherlands.csv")


# Convert Period to datetime, handling both year and month-year formats
def convert_period(period):
    try:
        return pd.to_datetime(str(period), format='%Y')
    except:
        return pd.to_datetime(period + ' 1', format='%Y %B %d')


df['Period'] = df['Period'].apply(convert_period)

# Separate annual and monthly data
annual_data = df[df['Period'].dt.year < 2024].copy()
monthly_data_2024 = df[df['Period'].dt.year == 2024].copy()

# Calculate 2024 projection using average of available monthly data
projected_2024 = monthly_data_2024['Annual rate of change'].mean()

# Calculate compounded inflation (starting at 0)
annual_data['Compounded_Value'] = (1 + annual_data['Annual rate of change'] / 100).cumprod() - 1
projected_compound = (annual_data['Compounded_Value'].iloc[-1] + 1) * (1 + projected_2024 / 100) - 1

# Define recession periods and their descriptions
recessions = [
    {
        'start': '1974',
        'end': '1975',
        'name': 'Oil Crisis\nRecession'
    },
    {
        'start': '1980',
        'end': '1983',
        'name': '80s Crisis'
    },
    {
        'start': '2001',
        'end': '2003',
        'name': 'Dot-com\nBubble'
    },
    {
        'start': '2008',
        'end': '2009',
        'name': 'Financial\nCrisis'
    },
    {
        'start': '2020',
        'end': '2023',
        'name': 'COVID-19\nPandemic'
    }
]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))


# Function to add recession highlights and calculate increases
def add_recession_highlights(ax, data_column='Annual rate of change'):
    for recession in recessions:
        start = pd.to_datetime(recession['start'])
        end = pd.to_datetime(recession['end'])
        ax.axvspan(start, end, color='lightblue', alpha=0.3)

        # Calculate position for recession label (centered in the recession period)
        mid_point = start + (end - start) / 2

        # Get y-axis limits
        ymin, ymax = ax.get_ylim()
        label_y = ymax * 0.9  # Position label high in the chart

        # Calculate total increase during recession
        start_value = annual_data.loc[annual_data['Period'].dt.year == int(recession['start']), data_column].iloc[0]
        end_value = annual_data.loc[annual_data['Period'].dt.year == int(recession['end']), data_column].iloc[0]

        if data_column == 'Compounded_Value':
            increase = f"+{(end_value - start_value) * 100:.1f}%"
        else:
            increase = f"Δ {end_value - start_value:.1f}%"

        # Add recession name and increase
        ax.text(mid_point, label_y, f"{recession['name']}\n{increase}",
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8)


# Plot 1: Annual Inflation Rate
ax1.plot(annual_data['Period'], annual_data['Annual rate of change'],
         marker='o', label='Historical Inflation')

# Add 2024 projection point
projection_date = pd.to_datetime('2024')
ax1.plot(projection_date, projected_2024,
         marker='o', color='red', markersize=10,
         label='Projection')

ax1.annotate(f'{projected_2024:.1f}%',
             xy=(projection_date, projected_2024),
             xytext=(10, 10),
             textcoords='offset points',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

ax1.set_title('Netherlands Annual Inflation Rate (1963-2024)', pad=20)
ax1.set_xlabel('Year')
ax1.set_ylabel('Annual Rate of Change (%)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# Add recession highlights to first plot
add_recession_highlights(ax1)

# Rotate x-axis labels for better readability
ax1.tick_params(axis='x', rotation=45)

# Add annotations for key events
highest_inflation = annual_data.loc[annual_data['Annual rate of change'].idxmax()]
lowest_inflation = annual_data.loc[annual_data['Annual rate of change'].idxmin()]

ax1.annotate(f'Highest: {highest_inflation["Annual rate of change"]}% ({highest_inflation["Period"].year})',
             xy=(highest_inflation['Period'], highest_inflation['Annual rate of change']),
             xytext=(10, 10), textcoords='offset points')

ax1.annotate(f'Lowest: {lowest_inflation["Annual rate of change"]}% ({lowest_inflation["Period"].year})',
             xy=(lowest_inflation['Period'], lowest_inflation['Annual rate of change']),
             xytext=(10, -10), textcoords='offset points')

# Plot 2: Prepare data for regression before plotting
# Convert dates to numbers for regression
x_dates = annual_data['Period'].values.astype(float)
y_values = annual_data['Compounded_Value'].values * 100

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x_dates, y_values)

# Create regression line points
x_reg = np.array([x_dates[0], x_dates[-1]])
y_reg = slope * x_reg + intercept

# Convert x_reg back to datetime for plotting
x_reg_dates = pd.to_datetime(x_reg.astype('datetime64[ns]'))

# Plot the data points
ax2.plot(annual_data['Period'], annual_data['Compounded_Value'] * 100,
         marker='o', label='Cumulative Increase')

# Plot regression line
ax2.plot(x_reg_dates, y_reg, '--',
         color='purple',
         linewidth=2.5,
         label=f'Linear Trend (R² = {r_value ** 2:.3f})')

# Add 2024 projection point
ax2.plot(projection_date, projected_compound * 100,
         marker='o', color='red', markersize=10,
         label='Projection')

ax2.annotate(f'{projected_compound * 100:.1f}%',
             xy=(projection_date, projected_compound * 100),
             xytext=(10, 10),
             textcoords='offset points',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

ax2.set_title('Cumulative Price Increase Since 1963', pad=20)
ax2.set_xlabel('Year')
ax2.set_ylabel('Total Price Increase (%)')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

# Add recession highlights to second plot with compound values
add_recession_highlights(ax2, 'Compounded_Value')

# Rotate x-axis labels for better readability
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()

# Calculate key statistics
stats_dict = {
    'Mean Inflation (1963-2023)': annual_data['Annual rate of change'].mean(),
    'Median Inflation (1963-2023)': annual_data['Annual rate of change'].median(),
    'Standard Deviation': annual_data['Annual rate of change'].std(),
    'Highest Annual Inflation': highest_inflation['Annual rate of change'],
    'Lowest Annual Inflation': lowest_inflation['Annual rate of change'],
    'Projected 2024 Inflation': projected_2024,
    'Total Price Increase (1963-2023)': annual_data['Compounded_Value'].iloc[-1] * 100,
    'Projected Total Increase by 2024': projected_compound * 100,
    'Average Annual Compound Rate': ((projected_compound + 1) ** (1 / 61) - 1) * 100,
    'Linear Trend Slope (% per year)': slope * 31557600,  # Convert seconds to years
    'R-squared': r_value ** 2
}

# Print analysis
print("\nNetherlands Inflation Analysis:")
print("-" * 50)
for key, value in stats_dict.items():
    print(f"{key}: {value:.2f}%")

# Create decade averages
annual_data['Decade'] = (annual_data['Period'].dt.year // 10) * 10
decade_avg = annual_data.groupby('Decade')['Annual rate of change'].mean()

print("\nDecade Averages:")
print("-" * 50)
for decade, avg in decade_avg.items():
    print(f"{decade}s: {avg:.2f}%")

plt.show()