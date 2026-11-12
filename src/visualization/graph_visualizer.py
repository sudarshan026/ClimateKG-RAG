import html
from typing import Dict, Any, List, Optional
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network

from src.knowledge_graph.schema import NodeType

# Professional color palette for Climate Knowledge Graph entities
NODE_COLOR_MAP: Dict[str, str] = {
    NodeType.CLIMATE_PHENOMENON.value: "#FF5722",   # Deep Orange
    NodeType.GREENHOUSE_GAS.value: "#D32F2F",       # Crimson Red
    NodeType.CLIMATE_IMPACT.value: "#7B1FA2",       # Royal Purple
    NodeType.REGION.value: "#1976D2",               # Ocean Blue
    NodeType.MITIGATION_STRATEGY.value: "#388E3C", # Emerald Green
    NodeType.ADAPTATION_STRATEGY.value: "#00796B", # Teal
    NodeType.POLICY.value: "#FBC02D",              # Amber Gold
    NodeType.ORGANIZATION.value: "#455A64",        # Blue Grey
    NodeType.RESEARCH_PAPER.value: "#5D4037",       # Brown/Bronze
    NodeType.DATASET.value: "#0097A7"              # Dark Cyan
}

class GraphVisualizer:
    """
    Renders interactive Knowledge Graph visualizations using PyVis and Plotly.
    """

    @staticmethod
    def generate_pyvis_html(
        graph: nx.MultiDiGraph,
        height: str = "600px",
        width: str = "100%",
        max_nodes: int = 150
    ) -> str:
        """
        Builds an interactive PyVis physics network simulation.
        Returns HTML string suitable for Streamlit components.v1.html embedding.
        """
        net = Network(height=height, width=width, directed=True, bgcolor="#0e1117", font_color="#e0e0e0")
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120, spring_strength=0.04)

        # Select top nodes by degree if graph exceeds max_nodes
        if graph.number_of_nodes() > max_nodes:
            deg = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
            sub_nodes = [n[0] for n in deg]
            view_graph = graph.subgraph(sub_nodes)
        else:
            view_graph = graph

        for nid, data in view_graph.nodes(data=True):
            name = data.get("name", nid)
            etype = data.get("type", "Entity")
            color = NODE_COLOR_MAP.get(etype, "#9E9E9E")
            title_tooltip = f"<b>{html.escape(name)}</b><br>Type: {html.escape(etype)}<br>ID: {html.escape(nid)}"

            net.add_node(
                nid,
                label=name,
                title=title_tooltip,
                color=color,
                size=22,
                font={"size": 13, "color": "#ffffff", "face": "Helvetica"}
            )

        for u, v, key, data in view_graph.edges(keys=True, data=True):
            rtype = data.get("type", "RELATED_TO")
            evidence = data.get("evidence_text", "")
            tooltip = f"<b>{rtype}</b><br>Evidence: {html.escape(evidence[:200])}"
            net.add_edge(
                u,
                v,
                label=rtype,
                title=tooltip,
                color="#78909C",
                arrows="to",
                width=1.5,
                font={"size": 9, "color": "#B0BEC5", "align": "middle"}
            )

        html_content = net.generate_html()
        return html_content

    @staticmethod
    def generate_plotly_network(
        graph: nx.MultiDiGraph,
        max_nodes: int = 100
    ) -> go.Figure:
        """
        Builds a native Plotly 2D interactive network figure.
        """
        if graph.number_of_nodes() > max_nodes:
            deg = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
            sub_nodes = [n[0] for n in deg]
            work_graph = graph.subgraph(sub_nodes)
        else:
            work_graph = graph

        pos = nx.spring_layout(work_graph.to_undirected(), seed=42, k=0.5)

        edge_x, edge_y = [], []
        for u, v in work_graph.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.0, color="#607D8B"),
            hoverinfo="none",
            mode="lines"
        )

        node_x, node_y, node_colors, node_text, node_names = [], [], [], [], []
        for node in work_graph.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                data = work_graph.nodes[node]
                name = data.get("name", node)
                etype = data.get("type", "Entity")
                node_colors.append(NODE_COLOR_MAP.get(etype, "#9E9E9E"))
                node_names.append(name)
                node_text.append(f"{name} ({etype})")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_names,
            textposition="top center",
            textfont=dict(size=9, color="#E0E0E0"),
            hovertext=node_text,
            marker=dict(
                size=18,
                color=node_colors,
                line=dict(width=2, color="#FFFFFF")
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=10, l=10, r=10, t=25),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117"
            )
        )
        return fig

