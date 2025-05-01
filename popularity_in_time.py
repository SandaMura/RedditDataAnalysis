import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('data/reddit_links.tsv', sep='\t')

# Convert 'TIMESTAMP' column to datetime
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# Create a new column for the week (starts on Monday by default)
df['week'] = df['TIMESTAMP'].dt.to_period('W').apply(lambda r: r.start_time)

# Count number of posts per subreddit per week
weekly_counts = df.groupby(['SOURCE_SUBREDDIT', 'week']).size().unstack(fill_value=0)

# Optional: Keep only the top 20 most active subreddits for clarity
top_subs = df['SOURCE_SUBREDDIT'].value_counts().head(20).index
weekly_counts = weekly_counts.loc[top_subs]

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(weekly_counts, cmap='YlGnBu', linewidths=0.5)
plt.title('Reddit Subreddit Popularity Over Time (Weekly)')
plt.xlabel('Week')
plt.ylabel('Subreddit')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
