import networkx as nx

# Load the GEXF graph
input_file = "data/reddit_hyperlink_graph.gexf"
G = nx.read_gexf(input_file)

# Subreddit to filter
target_subreddit = "funny"

# Filter edges where 'funny' is source or target
edges = [(u, v, d) for u, v, d in G.edges(data=True) if u == target_subreddit or v == target_subreddit]

# Create a subgraph with these edges
G_sub = nx.DiGraph()
G_sub.add_edges_from(edges)

# Ensure all connected nodes are included
nodes = set()
for u, v in G_sub.edges():
    nodes.add(u)
    nodes.add(v)
G_sub.add_nodes_from(nodes)

# Save the filtered subgraph
output_file = f"subreddit_{target_subreddit}_graph.gexf"
nx.write_gexf(G_sub, output_file)

print(f"Filtered graph for '{target_subreddit}' saved as '{output_file}'")
