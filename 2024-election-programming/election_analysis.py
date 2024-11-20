import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_election_returns(returns_file):
    """Calculate 3-month returns before elections using cleaned monthly returns data"""
    # Read returns data
    df = pd.read_csv(returns_file, parse_dates=['Date'])

    # Election years from 1928 onwards
    election_years = [1928, 1932, 1936, 1940, 1944, 1948, 1952, 1956, 1960, 1964,
                      1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004,
                      2008, 2012, 2016, 2020]

    election_returns = []

    for year in election_years:
        # Get August through October returns
        year_data = df[df['Date'].dt.year == year]
        aug_oct_returns = year_data[year_data['Date'].dt.month.isin([8, 9, 10])]

        # Calculate cumulative 3-month return
        cumulative_return = (1 + aug_oct_returns['Monthly_Return']).prod() - 1
        election_returns.append(cumulative_return * 100)  # Convert to percentage

    return election_returns


def create_historical_data(returns_file):
    """
    Creates a dataset based on verified historical S&P 500 returns and actual election outcomes.
    """
    # Get verified returns from monthly data for historical elections
    historical_returns = calculate_election_returns(returns_file)

    # Add 2024 prediction return
    returns = historical_returns + [3.35]  # Add 2024 prediction

    elections = {
        'year': [1928, 1932, 1936, 1940, 1944, 1948, 1952, 1956, 1960, 1964,
                 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004,
                 2008, 2012, 2016, 2020, 2024],
        'incumbent_party': ['R', 'R', 'D', 'D', 'D', 'D', 'D', 'R', 'R', 'D',
                            'D', 'R', 'R', 'D', 'R', 'R', 'R', 'D', 'D', 'R',
                            'R', 'D', 'D', 'R', 'D'],
        'candidate': ['Hoover', 'Hoover*', 'Roosevelt*', 'Roosevelt*', 'Roosevelt*', 'Truman*',
                      'Stevenson', 'Eisenhower*', 'Nixon', 'Johnson*', 'Humphrey', 'Nixon*',
                      'Ford*', 'Carter*', 'Reagan*', 'Bush', 'Bush*', 'Clinton*', 'Gore',
                      'Bush*', 'McCain', 'Obama*', 'Clinton', 'Trump*', 'Harris'],
        'opponent': ['Smith', 'Roosevelt', 'Landon', 'Willkie', 'Dewey', 'Dewey',
                     'Eisenhower', 'Stevenson', 'Kennedy', 'Goldwater', 'Nixon', 'McGovern',
                     'Carter', 'Reagan', 'Mondale', 'Dukakis', 'Clinton', 'Dole', 'G.W.Bush',
                     'Kerry', 'Obama', 'Romney', 'Trump', 'Biden', 'Trump'],
        'incumbent_running': [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        'party_won': [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, np.nan],
        'market_return_3m': returns,
        'market_direction_3m': ['UP' if r > 0 else 'DOWN' for r in returns]
    }

    df = pd.DataFrame(elections)

    # Add note about candidate type
    df['candidate'] = df['candidate'].astype(str) + df['incumbent_running'].map({1: '*', 0: ''})

    return df


def visualize_results(df):
    """Creates enhanced visualization of the market returns vs election outcomes"""
    plt.figure(figsize=(15, 10))

    # Split historical and prediction data
    historical_df = df[df['year'] < 2024].copy()
    prediction_df = df[df['year'] == 2024].copy()

    # Create base scatter plot for historical data
    sns.scatterplot(data=historical_df, x='year', y='market_return_3m',
                    s=150, alpha=0.6)

    # Add horizontal line at 0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Customize plot for historical points
    for idx, row in historical_df.iterrows():
        color = 'green' if row['party_won'] == 1 else 'red'
        marker = '^' if row['incumbent_party'] == 'D' else 'v'

        plt.scatter(row['year'], row['market_return_3m'],
                    color=color, marker=marker, s=150,
                    label=f"{row['incumbent_party']} {'Won' if row['party_won'] == 1 else 'Lost'}")

        # Add election matchup to the label
        label = f"{row['candidate']} vs {row['opponent']}\n{row['market_return_3m']:.1f}%"

        plt.annotate(label,
                     (row['year'], row['market_return_3m']),
                     xytext=(0, 10 if row['market_return_3m'] > 0 else -20),
                     textcoords='offset points',
                     ha='center',
                     fontsize=8)

    # Add 2024 prediction point
    if not prediction_df.empty:
        row = prediction_df.iloc[0]
        marker = '*' if row['incumbent_party'] == 'D' else 'v'
        plt.scatter(row['year'], row['market_return_3m'],
                    color='blue', marker=marker, s=200, alpha=0.7)

        # Add label for prediction point
        plt.annotate(f"Market 3.35% UP \n Harris vs Trump",
                     (row['year'], row['market_return_3m']),
                     xytext=(0, 15),
                     textcoords='offset points',
                     ha='center',
                     fontsize=10,
                     bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))

    # Calculate probabilities
    up_markets = df[df['market_direction_3m'] == 'UP']
    down_markets = df[df['market_direction_3m'] == 'DOWN']

    up_win_prob = (up_markets['party_won'] == 1).mean() * 100
    down_loss_prob = (down_markets['party_won'] == 0).mean() * 100

    # Calculate probabilities for incumbent running vs not running
    inc_df = df[df['incumbent_running'] == 1]
    non_inc_df = df[df['incumbent_running'] == 0]

    inc_up = inc_df[inc_df['market_direction_3m'] == 'UP']
    inc_down = inc_df[inc_df['market_direction_3m'] == 'DOWN']
    non_inc_up = non_inc_df[non_inc_df['market_direction_3m'] == 'UP']
    non_inc_down = non_inc_df[non_inc_df['market_direction_3m'] == 'DOWN']

    # Add probability information to the plot
    plt.text(0.02, 0.32,
             f"Historical Pattern (1928-2020):\n" +
             f"Overall Pattern:\n" +
             f"  Market UP ({len(up_markets)} cases): {up_win_prob:.1f}% chance incumbent party wins\n" +
             f"  Market DOWN ({len(down_markets)} cases): {down_loss_prob:.1f}% chance incumbent party loses\n\n" +
             f"When Incumbent Running ({len(inc_df)} cases):\n" +
             f"  UP ({len(inc_up)} cases): {(inc_up['party_won'] == 1).mean() * 100:.1f}% win rate\n" +
             f"  DOWN ({len(inc_down)} cases): {(inc_down['party_won'] == 0).mean() * 100:.1f}% loss rate\n\n" +
             f"New Candidate ({len(non_inc_df)} cases):\n" +
             f"  UP ({len(non_inc_up)} cases): {(non_inc_up['party_won'] == 1).mean() * 100:.1f}% win rate\n" +
             f"  DOWN ({len(non_inc_down)} cases): {(non_inc_down['party_won'] == 0).mean() * 100:.1f}% loss rate\n\n" +
             "* President running for re-election",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.title('3-Month S&P 500 Returns vs. Incumbent Party Success (1928-2020)\n(△ = Democrat, ▽ = Republican)',
              pad=20)
    plt.xlabel('Election Year')
    plt.ylabel('3-Month Return (Aug-Nov cumulative, %)')

    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               title="Historical Outcomes",
               bbox_to_anchor=(0.87, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def main():
    # File path
    returns_file = r'Cleaned Monthly Returns SP500 1923-2021.csv'

    # Create dataset with verified returns
    df = create_historical_data(returns_file)

    # Print detailed analysis
    print("\nDetailed Election Analysis (1928-2020):")
    print("-" * 50)
    for _, row in df.iterrows():
        print(f"{row['year']}: {row['candidate']} vs {row['opponent']}")
        print(f"3-month return: {row['market_return_3m']:.1f}%")
        print(f"Outcome: Incumbent party {'Won' if row['party_won'] else 'Lost'}")
        print("-" * 30)

    # Calculate overall probabilities
    up_markets = df[df['market_direction_3m'] == 'UP']
    down_markets = df[df['market_direction_3m'] == 'DOWN']

    print("\nOverall Probability Analysis:")
    print(
        f"UP Markets ({len(up_markets)} cases): {(up_markets['party_won'] == 1).mean() * 100:.1f}% chance incumbent party wins")
    print(
        f"DOWN Markets ({len(down_markets)} cases): {(down_markets['party_won'] == 0).mean() * 100:.1f}% chance incumbent party loses")

    # Calculate probabilities for incumbent vs non-incumbent
    inc_df = df[df['incumbent_running'] == 1]
    non_inc_df = df[df['incumbent_running'] == 0]

    print("\nIncumbent Running Analysis:")
    inc_up = inc_df[inc_df['market_direction_3m'] == 'UP']
    inc_down = inc_df[inc_df['market_direction_3m'] == 'DOWN']
    print(f"UP Markets ({len(inc_up)} cases): {(inc_up['party_won'] == 1).mean() * 100:.1f}% win rate")
    print(f"DOWN Markets ({len(inc_down)} cases): {(inc_down['party_won'] == 0).mean() * 100:.1f}% loss rate")

    print("\nNew Candidate Analysis:")
    non_inc_up = non_inc_df[non_inc_df['market_direction_3m'] == 'UP']
    non_inc_down = non_inc_df[non_inc_df['market_direction_3m'] == 'DOWN']
    print(f"UP Markets ({len(non_inc_up)} cases): {(non_inc_up['party_won'] == 1).mean() * 100:.1f}% win rate")
    print(f"DOWN Markets ({len(non_inc_down)} cases): {(non_inc_down['party_won'] == 0).mean() * 100:.1f}% loss rate")

    # Create visualization
    visualize_results(df)


if __name__ == "__main__":
    main()