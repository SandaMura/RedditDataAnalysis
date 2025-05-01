import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('data/reddit_links.tsv', sep='\t')

# Convert 'TIMESTAMP' column to datetime
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# Set TIMESTAMP as index
df.set_index('TIMESTAMP', inplace=True)


# Filter both subreddits
donald_counts = df[df['SOURCE_SUBREDDIT'] == 'the_donald'].resample('W').size()
conspiracy_counts = df[df['SOURCE_SUBREDDIT'] == 'conspiracy'].resample('W').size()

# Align both series
combined = pd.concat([donald_counts, conspiracy_counts], axis=1, keys=['the_donald', 'conspiracy']).fillna(0)

# Plot time series
combined.plot(figsize=(14, 6))
plt.title("Weekly Post Volume: r/the_donald vs r/conspiracy")
plt.xlabel("Date")
plt.ylabel("Post Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# Cross-correlation
from scipy.signal import correlate

# Normalize
donald_norm = (combined['the_donald'] - combined['the_donald'].mean()) / combined['the_donald'].std()
conspiracy_norm = (combined['conspiracy'] - combined['conspiracy'].mean()) / combined['conspiracy'].std()

# Compute cross-correlation
corr = correlate(donald_norm, conspiracy_norm, mode='full')
lags = range(-len(donald_norm) + 1, len(donald_norm))

# Plot cross-correlation
plt.figure(figsize=(12, 5))
plt.plot(lags, corr)
plt.title("Cross-Correlation: r/the_donald vs r/conspiracy")
plt.xlabel("Lag (weeks)")
plt.ylabel("Correlation")
plt.grid(True)
plt.axvline(0, color='red', linestyle='--')
plt.tight_layout()
plt.show()
