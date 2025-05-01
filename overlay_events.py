import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('data/reddit_links.tsv', sep='\t')

# Convert 'TIMESTAMP' column to datetime
#df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])


# Assuming df is your main DataFrame with 'TIMESTAMP' and 'subreddit' columns
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df.set_index('TIMESTAMP', inplace=True)

# Filter only 'the_donald'
donald_df = df[df['SOURCE_SUBREDDIT'] == 'the_donald']

# Resample weekly
weekly_counts = donald_df.resample('W').size()

# Key political events to overlay
events = {
    'Trump announces campaign': '2015-06-16',
    'Republican nomination': '2016-07-19',
    'US Election': '2016-11-08',
    'Inauguration': '2017-01-20',
    'Subreddit quarantined': '2019-06-26',
    'Subreddit banned': '2020-06-29',
}

# Plotting
plt.figure(figsize=(14, 6))
weekly_counts.plot(label='Post Volume', color='darkred')

# Overlay events
for event, date in events.items():
    plt.axvline(pd.to_datetime(date), color='gray', linestyle='--', linewidth=1)
    plt.text(pd.to_datetime(date), plt.ylim()[1]*0.8, event, rotation=90, fontsize=8)

plt.title("Post Activity in r/the_donald with Key Political Events")
plt.xlabel("Date")
plt.ylabel("Weekly Post Count")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
