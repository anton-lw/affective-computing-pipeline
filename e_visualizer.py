# e_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_wellbeing_trajectory(signal, change_points):
    """Plots the main well-being trajectory with detected change points."""
    print("\n--- Generating well-being trajectory plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 8))
    
    ax.plot(signal.index, signal.values, color='cornflowerblue', linewidth=1.5, label='30-day sentiment rolling avg')
    ax.set_title('Well-being trajectory', fontsize=18)
    ax.set_ylabel('Sentiment score')
    
    for i, cp_date in enumerate(change_points):
        label = 'Significant change point' if i == 0 else None
        ax.axvline(cp_date, color='red', linestyle='--', alpha=0.6, label=label)
    
    ax.legend()

def plot_multivariate_markers(daily_data):
    """Plots normalized trajectories of multiple validated psychological markers."""
    print("\n--- Generating multivariate marker plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Normalize data for plotting on the same scale (z-score)
    for col in ['vader_compound_mean', 'i_talk_freq_mean', 'absolutist_freq_mean']:
        if col in daily_data.columns:
            rolling_col = daily_data[col].rolling(window=30).mean()
            z_score_col = (rolling_col - rolling_col.mean()) / rolling_col.std()
            
            label_map = {
                'vader_compound_mean': 'Sentiment (VADER)',
                'i_talk_freq_mean': 'Self-focus ("I-talk")',
                'absolutist_freq_mean': 'Absolutist thinking'
            }
            color_map = {
                'vader_compound_mean': 'blue',
                'i_talk_freq_mean': 'red',
                'absolutist_freq_mean': 'green'
            }
            linestyle_map = {
                'vader_compound_mean': '-',
                'i_talk_freq_mean': '--',
                'absolutist_freq_mean': ':'
            }
            ax.plot(z_score_col.index, z_score_col.values, label=label_map[col], 
                    color=color_map[col], linestyle=linestyle_map[col], linewidth=2, alpha=0.8)

    ax.set_title('Normalized well-being markers over time', fontsize=18)
    ax.set_ylabel('Standard deviations from mean (z-score)')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.legend(fontsize=12)

def plot_perma_sentiment(df):
    """Plots the average sentiment associated with each PERMA pillar."""
    print("\n--- Generating PERMA pillar sentiment plot ---")
    if 'perma_labels' not in df.columns or df[df['perma_labels'] != 'Not Processed']['perma_labels'].isnull().all():
        print("No PERMA data available to plot.")
        return

    perma_df = df[~df['perma_labels'].isin(['Not Processed', 'Error'])].copy()
    perma_df['perma_labels'] = perma_df['perma_labels'].str.split(r',\s*')
    perma_df = perma_df.explode('perma_labels')
    
    topic_sentiment = perma_df.groupby('perma_labels')['vader_compound'].mean().sort_values(ascending=False)
    
    if not topic_sentiment.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(x=topic_sentiment.values, y=topic_sentiment.index, palette='viridis', ax=ax)
        ax.set_title('Average sentiment by PERMA pillar', fontsize=16)
        ax.set_xlabel('Mean VADER compound score')
        ax.set_ylabel('PERMA pillar')
        ax.axvline(0, color='black', linewidth=0.5)

def show_plots():
    """Displays all generated plots."""
    print("\nDisplaying all generated plots. Close the plot windows to exit.")
    plt.show()