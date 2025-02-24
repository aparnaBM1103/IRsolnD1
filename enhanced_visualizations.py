import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Optional

class EnhancedVisualizations:
    def __init__(self):
        # SDG colors from official SDG branding guidelines
        self.sdg_colors = {
            'SDG1': '#E5243B', 'SDG2': '#DDA63A', 'SDG3': '#4C9F38',
            'SDG4': '#C5192D', 'SDG5': '#FF3A21', 'SDG6': '#26BDE2',
            'SDG7': '#FCC30B', 'SDG8': '#A21942', 'SDG9': '#FD6925',
            'SDG10': '#DD1367', 'SDG11': '#FD9D24', 'SDG12': '#BF8B2E',
            'SDG13': '#3F7E44', 'SDG14': '#0A97D9', 'SDG15': '#56C02B',
            'SDG16': '#00689D', 'SDG17': '#19486A'
        }

        # Theme colors
        self.theme_colors = {
            'Climate Change Mitigation': '#1f77b4',
            'Financial Inclusion': '#2ca02c',
            'Agriculture': '#ff7f0e',
            'Education': '#9467bd',
            'Healthcare': '#d62728'
        }

    def create_portfolio_treemap(self, df: pd.DataFrame, path: List[str], values: str) -> go.Figure:
        """Create enhanced treemap visualization"""
        fig = px.treemap(
            df,
            path=path,
            values=values,
            color_discrete_map=self.theme_colors
        )
        
        fig.update_layout(
            margin=dict(t=30, l=10, r=10, b=10),
            font=dict(size=12)
        )
        
        return fig

    def create_sdg_breakdown(self, df: pd.DataFrame) -> go.Figure:
        """Create SDG breakdown visualization"""
        sdg_summary = df.groupby(['sdg', 'impact_theme'])['capital_invested'].sum().reset_index()
        
        fig = px.sunburst(
            sdg_summary,
            path=['sdg', 'impact_theme'],
            values='capital_invested',
            color='sdg',
            color_discrete_map=self.sdg_colors
        )
        
        fig.update_layout(
            margin=dict(t=30, l=10, r=10, b=10),
            font=dict(size=12)
        )
        
        return fig

    def create_progress_chart(self, row: pd.Series) -> go.Figure:
        fig = go.Figure()
        
        years = ['2021', '2022', '2023']
        progress = [row[f'progress_year{i}'] for i in range(1, 4)]
        
        fig.add_trace(go.Scatter(
            x=years,
            y=progress,
            mode='lines+markers',
            name='Progress'
        ))
        
        fig.update_layout(
            height=100,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            xaxis=dict(
                tickmode='array',
                ticktext=years,
                tickvals=years
            )
        )
        
        return fig

    def create_relevance_indicator(self, relevance: str) -> Dict:
        """Create relevance indicator style"""
        colors = {
            'High': '#3f7e44',
            'Medium': '#fcc30b',
            'Low': '#e5243b'
        }
        
        return {
            'width': '24px',
            'height': '24px',
            'border-radius': '12px',
            'background-color': colors.get(relevance, '#808080'),
            'margin': '0 auto'
        }

    def create_benchmark_gauge(self, 
                             current: float,
                             benchmark: float,
                             title: str = "Performance vs Benchmark") -> go.Figure:
        """Create enhanced benchmark gauge"""
        fig = go.Figure(go.Indicator(
            mode="gauge+delta",
            value=current,
            delta={'reference': benchmark},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, benchmark], 'color': "lightgray"},
                    {'range': [benchmark, 100], 'color': "#e8f5e9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': benchmark
                }
            },
            title={'text': title}
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(t=30, l=10, r=10, b=10)
        )
        
        return fig

    def create_metrics_coverage(self, df: pd.DataFrame) -> go.Figure:
        """Create metrics coverage heatmap"""
        coverage_data = df.groupby(['impact_theme', 'impact_outcome']).agg({
            'company': 'nunique',
            'impact_kpi': 'nunique'
        }).reset_index()
        
        fig = px.density_heatmap(
            coverage_data,
            x='impact_theme',
            y='impact_outcome',
            z='company',
            color_continuous_scale=[
                [0, '#f7fbff'],
                [0.5, '#6baed6'],
                [1, '#08519c']
            ],
            labels={'company': 'Companies'}
        )
        
        fig.update_layout(
            title='Impact Coverage Matrix',
            xaxis_title='Impact Theme',
            yaxis_title='Impact Outcome',
            height=400,
            margin=dict(t=30, l=10, r=10, b=10)
        )
        
        return fig

    def create_performance_summary(self, df: pd.DataFrame) -> go.Figure:
        """Create performance summary visualization"""
        summary = df.groupby(['impact_theme', 'impact_outcome']).agg({
            'progress_year3': lambda x: (x / df['impact_target']) * 100,
            'impact_target': 'mean',
            'company': 'count'
        }).reset_index()
        
        fig = px.scatter(
            summary,
            x='impact_theme',
            y='progress_year3',
            size='company',
            color='impact_outcome',
            hover_data=['impact_target'],
            color_discrete_map=self.theme_colors
        )
        
        fig.update_layout(
            height=400,
            yaxis_title='Progress Towards Target (%)',
            showlegend=True
        )
        
        return fig

    def create_company_comparison(self, df: pd.DataFrame, metric: str) -> go.Figure:
        """Create company comparison chart"""
        latest = df.sort_values('date').groupby('company').last()
        
        fig = go.Figure()
        
        companies = latest.index
        values = latest[metric]
        targets = latest['impact_target']
        
        # Add bars for actual values
        fig.add_trace(go.Bar(
            x=companies,
            y=values,
            name='Current',
            marker_color='#1f77b4'
        ))
        
        # Add target markers
        fig.add_trace(go.Scatter(
            x=companies,
            y=targets,
            mode='markers',
            name='Target',
            marker=dict(symbol='diamond', size=12, color='red')
        ))
        
        fig.update_layout(
            barmode='group',
            height=300,
            margin=dict(t=30, l=10, r=10, b=10)
        )
        
        return fig

    def create_impact_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create impact distribution violin plot"""
        fig = go.Figure()
        
        for theme in df['impact_theme'].unique():
            theme_data = df[df['impact_theme'] == theme]
            
            fig.add_trace(go.Violin(
                x=theme_data['impact_theme'],
                y=theme_data['progress_year3'],
                name=theme,
                box_visible=True,
                meanline_visible=True,
                line_color=self.theme_colors.get(theme, '#1f77b4')
            ))
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_title='Progress Year 3',
            xaxis_title='Impact Theme'
        )
        
        return fig