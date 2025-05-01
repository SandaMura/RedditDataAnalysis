import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
file_path = 'data/reddit_links.tsv'
df = pd.read_csv(file_path, sep='\t')

# Convert timestamp column to datetime
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# Parse sentiment and properties into lists of floats
#df['LINK_SENTIMENT'] = df['LINK_SENTIMENT'].apply(lambda x: list(map(float, x.split(','))))
#df['PROPERTIES'] = df['PROPERTIES'].apply(lambda x: list(map(float, x.split(','))))

df['LINK_SENTIMENT'] = df['LINK_SENTIMENT'].apply(
    lambda x: list(map(float, str(x).split(','))) if isinstance(x, str) else [float(x)]
)

df['PROPERTIES'] = df['PROPERTIES'].apply(
    lambda x: list(map(float, str(x).split(','))) if isinstance(x, str) else [float(x)]
)




# Initialize directed graph
G = nx.DiGraph()

# Build the graph from the dataframe
for _, row in df.iterrows():
    src = row['SOURCE_SUBREDDIT']
    tgt = row['TARGET_SUBREDDIT']
    
    # Add node and edge
    G.add_edge(
        src, tgt,
        timestamp=row['TIMESTAMP'],
        sentiment=row['LINK_SENTIMENT'],
        properties=row['PROPERTIES']
    )

print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Save to GEXF file
def avg(lst):
    return sum(lst) / len(lst) if isinstance(lst, list) and lst else 0.0

for u, v, data in G.edges(data=True):
    # Convert timestamp to string
    if isinstance(data.get('timestamp'), pd.Timestamp):
        data['timestamp'] = data['timestamp'].isoformat()
    
    # Convert list attributes to average (or str if needed)
    if 'sentiment' in data:
        data['sentiment_avg'] = avg(data['sentiment'])  # float value
        del data['sentiment']  # remove the list version

    if 'properties' in data:
        data['properties_avg'] = avg(data['properties'])  # float value
        del data['properties']  # remove the list version
        
nx.write_gexf(G, "reddit_hyperlink_graph.gexf")

print("Graph exported successfully to 'reddit_hyperlink_graph.gexf'")

# Optional: Draw the graph (small sample)
if G.number_of_nodes() < 100:
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, with_labels=True, node_size=50, font_size=8, edge_color='gray')
    plt.title("Reddit Subreddit Hyperlink Graph")
    plt.show()
else:
    print("Graph too large to visualize directly. Try filtering or exporting.")
