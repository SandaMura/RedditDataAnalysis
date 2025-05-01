import networkx as nx
import matplotlib.pyplot as plt

# Load the filtered subreddit graph
input_file = "subreddit_funny_graph.gexf"
G = nx.read_gexf(input_file)

print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# Optional: for reproducible layout
pos = nx.spring_layout(G, seed=42)

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("Subreddit Interaction Graph: r/funny")
plt.axis('off')
plt.tight_layout()
plt.show()
