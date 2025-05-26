# %%
from __future__ import annotations

import math
import pathlib
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pyvis.network import Network
from upsetplot import UpSet, from_memberships

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
except ImportError:
    KaplanMeierFitter = CoxPHFitter = None

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    Levenshtein = None

# %%
ROOT = pathlib.Path(".").resolve().parent
GRAPHS_DIR = ROOT / "graphs"
PROC_DIR = ROOT / "processed"
FIG_DIR = ROOT / "figures" / "networks"
HTML_DIR = FIG_DIR / "html"

FIG_DIR.mkdir(exist_ok=True)
HTML_DIR.mkdir(exist_ok=True)

PLOTLY_TEMPL = "plotly_white"


# %%
def load_graph(name: str) -> nx.Graph:
    """Load a GraphML file and return a NetworkX graph."""
    path = GRAPHS_DIR / f"{name}.graphml"
    return nx.read_graphml(path)


def add_cluster_attribute(G: nx.Graph, resolution: float = 1.0, attr: str = "cluster"):
    """Add Louvain community IDs as a node attribute."""
    if community_louvain is None:
        raise ImportError("python-louvain is not installed in this environment.")
    partition = community_louvain.best_partition(nx.Graph(G), resolution=resolution)
    nx.set_node_attributes(G, partition, attr)
    return partition


# %%
def interactive_flow_map(
    html_out: pathlib.Path = HTML_DIR / "repost_flow.html", max_nodes: int | None = None
):
    """Interactive PyVis network for the repost-flow graph."""
    G = load_graph("repost_flow")
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        deg = dict(G.degree(weight="weight"))
        top_nodes = {
            n
            for n, _ in sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[
                :max_nodes
            ]
        }
        G = G.subgraph(top_nodes).copy()

    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=["physics"])
    net.save_graph(str(html_out))
    print(f"✔ Interactive flow map → {html_out}")


# %%
def inout_scatter(plot_out: pathlib.Path = HTML_DIR / "inout_scatter.html"):
    G = load_graph("repost_flow")
    add_cluster_attribute(G)
    df = pd.DataFrame(
        {
            "subreddit": list(G.nodes()),
            "in_degree": [G.in_degree(n, weight="weight") for n in G.nodes()],
            "out_degree": [G.out_degree(n, weight="weight") for n in G.nodes()],
            "strength": [G.degree(n, weight="weight") for n in G.nodes()],
            "cluster": [G.nodes[n]["cluster"] for n in G.nodes()],
        }
    )
    fig = px.scatter(
        df,
        x="out_degree",
        y="in_degree",
        color="cluster",
        size="strength",
        hover_name="subreddit",
        template=PLOTLY_TEMPL,
        height=700,
        title="In‑ vs Out‑degree",
    )
    fig.write_html(plot_out)
    print(f"✔ Scatter → {plot_out}")


# %%
def edge_weight_zipf(plot_out: pathlib.Path = FIG_DIR / "edge_weight_zipf.png"):
    G = load_graph("repost_flow")
    weights = sorted([d["weight"] for _, _, d in G.edges(data=True)], reverse=True)
    ranks = np.arange(1, len(weights) + 1)
    plt.figure(figsize=(6, 4))
    plt.loglog(ranks, weights, marker=".")
    plt.xlabel("Edge rank (log)")
    plt.ylabel("Edge weight (log)")
    plt.title("Zipf plot of edge weights")
    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()
    print(f"✔ Zipf plot → {plot_out}")


# %%
def sankey_top_images(
    n_images: int = 10, html_out: pathlib.Path = HTML_DIR / "sankey_images.html"
):
    df = pd.read_parquet(PROC_DIR / "submissions_final.parquet")
    top_imgs = (
        df.groupby("image_id").size().sort_values(ascending=False).head(n_images).index
    )
    records = []
    for img in top_imgs:
        hops = df[df.image_id == img].sort_values("unixtime")["subreddit"].tolist()[:3]
        if len(hops) == 3:
            records.append(hops)
    if not records:
        print("⚠ No images with ≥3 hops – skip Sankey.")
        return
    nodes = sorted({s for seq in records for s in seq})
    nidx = {s: i for i, s in enumerate(nodes)}
    src, dst, val = [], [], []
    for seq in records:
        for a, b in zip(seq, seq[1:]):
            src.append(nidx[a])
            dst.append(nidx[b])
            val.append(1)
    fig = go.Figure(
        go.Sankey(
            node=dict(label=nodes, pad=10), link=dict(source=src, target=dst, value=val)
        )
    )
    fig.update_layout(title="Repost paths for top images", template=PLOTLY_TEMPL)
    fig.write_html(html_out)
    print(f"✔ Sankey → {html_out}")


# %%
def chord_diagram(out=HTML_DIR / "chord_corepost.html", thr=50):
    G = load_graph("corepost_projection")
    E = [(u, v, d["weight"]) for u, v, d in G.edges(data=True) if d["weight"] >= thr]
    if not E:
        print("threshold too high – no edges")
        return
    labs = sorted({u for u, _, _ in E} | {v for _, v, _ in E})
    mat = np.zeros((n := len(labs), n))
    id = {l: i for i, l in enumerate(labs)}
    [
        (lambda i, j, w: (mat.__setitem__((i, j), w), mat.__setitem__((j, i), w)))(
            id[u], id[v], w
        )
        for u, v, w in E
    ]
    # try native Chord first (plotly ≥5.15)
    if hasattr(go, "Chord"):
        fig = go.Figure(go.Chord(labels=labs, matrix=mat.tolist()))
    else:  # fallback thin‑arc sankey
        src, dst, val = [], [], []
        [src.append(id[u]) or dst.append(id[v]) or val.append(w) for u, v, w in E]
        fig = go.Figure(
            go.Sankey(
                arrangement="fixed",
                node=dict(label=labs, pad=7, thickness=8),
                link=dict(source=src, target=dst, value=val),
            )
        )
    fig.update_layout(
        title=f"Co‑repost diagram (≥{thr} shared images)",
        template=PLOTLY_TEMPL,
        height=700,
    )
    fig.write_html(out)


# %%
def upset_corepost(
    plot_out: pathlib.Path = FIG_DIR / "upset_corepost.png", top_k: int = 12
):
    G = load_graph("corepost_projection")
    strength = {n: G.degree(n, weight="weight") for n in G.nodes()}
    top_subs = [
        n
        for n, _ in sorted(strength.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    ]
    imgdf = pd.read_parquet(PROC_DIR / "submissions_final.parquet")[
        ["image_id", "subreddit"]
    ]
    memberships = (
        imgdf[imgdf.subreddit.isin(top_subs)]
        .groupby("image_id")["subreddit"]
        .apply(list)
        .tolist()
    )
    upset = from_memberships(memberships)
    plt.figure(figsize=(10, 5))
    UpSet(upset, subset_size="count", show_counts=True).plot()
    plt.suptitle("UpSet – intersections across top subs")
    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()
    print(f"✔ UpSet → {plot_out}")


# %%
def blockmodel_heatmap(plot_out: pathlib.Path = FIG_DIR / "block_heatmap.png"):
    G = load_graph("corepost_projection")
    part = add_cluster_attribute(G)
    order = sorted(G.nodes(), key=lambda n: (part[n], G.degree(n, weight="weight")))
    idx = {n: i for i, n in enumerate(order)}
    mat = np.zeros((len(order), len(order)))
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        mat[i, j] = mat[j, i] = math.log1p(d["weight"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, cmap="mako_r", xticklabels=False, yticklabels=False)
    plt.title("Block‑model heat‑map (log1p weights)")
    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()
    print(f"✔ Heat‑map → {plot_out}")


# %%
def sunburst_communities(
    html_out: pathlib.Path = HTML_DIR / "sunburst_communities.html",
):
    G = load_graph("corepost_projection")
    partition = add_cluster_attribute(G)
    df = pd.DataFrame(
        {
            "subreddit": list(partition.keys()),
            "cluster": list(partition.values()),
            "strength": [G.degree(n, weight="weight") for n in partition],
        }
    )
    fig = px.sunburst(
        df,
        path=["cluster", "subreddit"],
        values="strength",
        template=PLOTLY_TEMPL,
        title="Community sunburst",
    )
    fig.write_html(html_out)
    print(f"✔ Sunburst → {html_out}")


# %%
def freq_vs_median_gain_scatter(out=HTML_DIR / "freq_vs_gain.html"):
    df = pd.DataFrame(
        [
            {**d, "src": u, "dst": v}
            for u, v, d in load_graph("repost_amplification").edges(data=True)
        ]
    )
    size_col = df.mean_gain.abs().add(1)  # ensure >0
    px.scatter(
        df,
        x="count",
        y="median_gain",
        size=size_col,
        color=(df.median_gain > 0),
        hover_data=["src", "dst"],
        template=PLOTLY_TEMPL,
        height=700,
        title="Edge freq vs median karma gain (size = |mean_gain|+1)",
    ).write_html(out)


# %%
def edge_rank_bump_chart(
    html_out: pathlib.Path = HTML_DIR / "edge_bump.html", top_k: int = 15
):
    # Approximate: rank subreddits by yearly median gain (source→any)
    df = pd.read_parquet(PROC_DIR / "submissions_final.parquet")
    df = df.sort_values(["image_id", "unixtime"])
    df["gain"] = df.groupby("image_id")["score"].diff()
    df = df.dropna(subset=["gain"])
    df["year"] = pd.to_datetime(df.unixtime, unit="s").dt.year
    g = df.groupby(["year", "subreddit"])["gain"].median().reset_index()
    g["rank"] = g.groupby("year")["gain"].rank(ascending=False, method="first")
    top = g[g["rank"] <= top_k]
    fig = px.line(
        top,
        x="year",
        y="rank",
        color="subreddit",
        hover_data=["gain"],
        template=PLOTLY_TEMPL,
        height=700,
        title="Median‑gain rank evolution (top edges)",
    )
    fig.update_yaxes(autorange="reversed")
    fig.write_html(html_out)
    print(f"✔ Bump chart → {html_out}")


# %%
def exporter_bar_race(
    html_out: pathlib.Path = HTML_DIR / "bar_race.html", period: str = "M"
):
    df = pd.read_parquet(PROC_DIR / "submissions_final.parquet")
    df = df.sort_values(["image_id", "unixtime"])
    df["gain"] = df.groupby("image_id")["score"].diff()
    df = df.dropna(subset=["gain"])
    df["period"] = (
        pd.to_datetime(df.unixtime, unit="s").dt.to_period(period).dt.to_timestamp()
    )
    gains = df.groupby(["period", "subreddit"])["gain"].mean().reset_index()
    gains["cum_gain"] = gains.groupby("subreddit")["gain"].cumsum()
    fig = px.bar(
        gains,
        x="cum_gain",
        y="subreddit",
        animation_frame="period",
        orientation="h",
        range_x=[0, gains.cum_gain.max() * 1.05],
        template=PLOTLY_TEMPL,
        height=700,
        title="Cumulative avg karma exported per subreddit",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.write_html(html_out)
    print(f"✔ Bar‑race → {html_out}")


# %%
def resubmission_hist(out=FIG_DIR / "resubmission_hist.png"):
    cnt = (
        pd.read_parquet(PROC_DIR / "submissions_final.parquet")
        .groupby("image_id")
        .size()
    )
    plt.figure(figsize=(6, 4))
    sns.histplot(cnt, bins=50, log_scale=(False, True))
    plt.xlabel("# resubmissions per image")
    plt.ylabel("Images (log)")
    plt.title("Distribution of image resubmissions")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# %%
ALL_FUNCS = [
    interactive_flow_map,
    inout_scatter,
    edge_weight_zipf,
    sankey_top_images,
    chord_diagram,
    upset_corepost,
    blockmodel_heatmap,
    sunburst_communities,
    freq_vs_median_gain_scatter,
    edge_rank_bump_chart,
    exporter_bar_race,
    resubmission_hist,
]


def run_all():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        start = datetime.now()
        for fn in ALL_FUNCS:
            fname = fn.__name__
            try:
                print(f"→ {fname}()")
                if fname == "sankey_top_images":
                    fn(n_images=10)
                else:
                    fn()
            except Exception as e:
                print(f"⚠ {fname} failed: {e}")
        print(
            f"Completed in {datetime.now() - start} – outputs in {FIG_DIR} & {HTML_DIR}"
        )


# %%
run_all()
