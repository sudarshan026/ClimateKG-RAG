from typing import List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

class MetricsVisualizer:
    """
    Produces publication-quality Plotly figures for IR benchmark evaluation comparisons.
    """

    @staticmethod
    def create_metrics_bar_chart(summary_df: pd.DataFrame) -> go.Figure:
        """Grouped bar chart comparing IR metrics across retrieval strategies."""
        metrics_cols = ["Precision@K", "Recall@K", "MRR", "MAP", "NDCG@K"]
        available_cols = [c for c in metrics_cols if c in summary_df.columns]

        fig = go.Figure()
        colors = ["#4285F4", "#34A853", "#FBBC05", "#EA4335", "#8E24AA"]

        for idx, col in enumerate(available_cols):
            fig.add_trace(go.Bar(
                name=col,
                x=summary_df["Retrieval Method"],
                y=summary_df[col],
                marker_color=colors[idx % len(colors)],
                text=summary_df[col].round(3),
                textposition="auto"
            ))

        fig.update_layout(
            barmode="group",
            title="Information Retrieval Benchmark Metrics Comparison",
            xaxis_title="Retrieval Method",
            yaxis_title="Score (0.0 to 1.0)",
            yaxis=dict(range=[0, 1.05]),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#E0E0E0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    def create_latency_bar_chart(summary_df: pd.DataFrame) -> go.Figure:
        """Bar chart comparing retrieval execution latency."""
        col = "Avg Latency (ms)"
        if col not in summary_df.columns:
            return go.Figure()

        fig = go.Figure(go.Bar(
            x=summary_df["Retrieval Method"],
            y=summary_df[col],
            text=summary_df[col].apply(lambda x: f"{x:.1f} ms"),
            textposition="auto",
            marker_color=["#26A69A", "#42A5F5", "#AB47BC", "#FFA726", "#FF7043"]
        ))

        fig.update_layout(
            title="Retrieval Latency Comparison (milliseconds)",
            xaxis_title="Retrieval Method",
            yaxis_title="Latency (ms)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#E0E0E0")
        )
        return fig

    @staticmethod
    def create_radar_chart(summary_df: pd.DataFrame) -> go.Figure:
        """Radar / Spider chart depicting overall multi-dimensional retrieval performance."""
        categories = ["Precision@K", "Recall@K", "MRR", "MAP", "NDCG@K"]
        available_cats = [c for c in categories if c in summary_df.columns]

        fig = go.Figure()

        for _, row in summary_df.iterrows():
            values = [row[c] for c in available_cats]
            # Close the polygon
            values.append(values[0])
            cats_closed = available_cats + [available_cats[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=cats_closed,
                fill="toself",
                name=row["Retrieval Method"]
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.0], color="#B0BEC5"),
                bgcolor="#161b22"
            ),
            title="Multi-Criteria Retrieval Radar Analysis",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#E0E0E0"),
            showlegend=True
        )
        return fig

