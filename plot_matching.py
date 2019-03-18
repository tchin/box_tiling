import sys
import networkx as nx

sys.path.insert(0, '../graph_drawing')
from plotly_visualize import visualize_graph_3d


def plot_3d_matching(tiling, filename="outputs/3dmatch.html"):
    visualize_graph_3d(nx.to_undirected(tiling), tiling.nodes(), [], filename, title="3d", coords={v: v for v in tiling.nodes()})