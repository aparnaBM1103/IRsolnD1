import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional
import io
import re

class ImpactDashboard:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="Impact Data Dashboard")
    
        # Custom CSS to reduce sidebar width
        st.markdown("""
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] {
                min-width: 200px;
                max-width: 200px;
            }
            
            /* Control multiselect dropdown size */
            div[data-baseweb="select"] > div {
                min-height: 38px;
                max-height: 38px;
            }
            
            /* Control height of the dropdown list when opened */
            ul[data-baseweb="menu"] {
                max-height: 200px !important;
            }
            
            /* Make filter section more compact */
            .stSelectbox, .stMultiSelect {
                margin-bottom: 0.5rem;
            }
            
            /* Adjust header spacing to make filter area more compact */
            .stMarkdown h2, .stMarkdown h3 {
                margin-top: 0.5rem;
                margin-bottom: 0.25rem;
            }
            </style>
            """, unsafe_allow_html=True)
        
        self.initialize_session_state()
        self.render_sidebar()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'page' not in st.session_state:
            st.session_state.page = 'dashboard'
            
        if 'data' not in st.session_state:
            st.session_state.data = pd.DataFrame()
            
        if 'drill_level' not in st.session_state:
            st.session_state.drill_level = 'portfolio'
            
        if 'selected_fund' not in st.session_state:
            st.session_state.selected_fund = None
            
        if 'selected_theme' not in st.session_state:
            st.session_state.selected_theme = None
            
        if 'upload_stage' not in st.session_state:
            st.session_state.upload_stage = 0
            
        if 'dimensions_status' not in st.session_state:
            st.session_state.dimensions_status = {}
            
        if 'filters' not in st.session_state:
            st.session_state.filters = {
                'selected_funds': [],
                'selected_themes': [],
                'selected_sdgs': []
            }
            
        if 'expanded_rows' not in st.session_state:
            st.session_state.expanded_rows = set()

    def render_sidebar(self):
        """Render navigation sidebar"""
        st.sidebar.title("Navigation")
        
        # Navigation buttons
        if st.sidebar.button("📥 Import Data"):
            st.session_state.page = 'import_data'
            st.session_state.upload_stage = 0
            
        if st.sidebar.button("📊 Dashboard"):
            st.session_state.page = 'dashboard'
            
        if st.sidebar.button("📈 Impact Analysis"):
            st.session_state.page = 'impact_analysis'
            
        if st.sidebar.button("📝 Impact Narrative"):
            st.session_state.page = 'impact_narrative'

    def render_dashboard_filters(self):
        """Render dashboard filters at the top of the page in a compact format"""
        st.markdown("### Filters")
        
        if not st.session_state.data.empty:
            # Use a container to keep things tight
            with st.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Fund filter with custom height
                    selected_funds = st.multiselect(
                        "Funds",  # Shorter label
                        options=sorted(st.session_state.data['fund'].unique()),
                        default=sorted(st.session_state.data['fund'].unique()),
                        key="filter_funds"
                    )
                
                with col2:
                    # Theme filter    
                    selected_themes = st.multiselect(
                        "Impact Themes",
                        options=sorted(st.session_state.data['impact_theme'].unique()),
                        default=sorted(st.session_state.data['impact_theme'].unique()),
                        key="filter_themes"
                    )
                
                with col3:
                    # SDG filter
                    selected_sdgs = st.multiselect(
                        "SDGs",  # Shorter label
                        options=sorted(st.session_state.data['sdg'].unique()),
                        default=sorted(st.session_state.data['sdg'].unique()),
                        key="filter_sdgs"
                    )
                
                st.session_state.filters = {
                    'selected_funds': selected_funds,
                    'selected_themes': selected_themes,
                    'selected_sdgs': selected_sdgs
                }
                
            # Add a subtle divider
            st.markdown("<hr style='margin: 0.5rem 0; border: none; height: 1px; background-color: #f0f0f0;'>", unsafe_allow_html=True)

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply dashboard filters to dataframe"""
        filtered_df = df.copy()
        
        if st.session_state.filters['selected_funds']:
            filtered_df = filtered_df[
                filtered_df['fund'].isin(st.session_state.filters['selected_funds'])
            ]
            
        if st.session_state.filters['selected_themes']:
            filtered_df = filtered_df[
                filtered_df['impact_theme'].isin(st.session_state.filters['selected_themes'])
            ]
            
        if st.session_state.filters['selected_sdgs']:
            filtered_df = filtered_df[
                filtered_df['sdg'].isin(st.session_state.filters['selected_sdgs'])
            ]
            
        return filtered_df
    
    def create_progress_chart(self, row: pd.Series) -> go.Figure:
        """Create progress chart for a row of data"""
        fig = go.Figure()
        
        # Progress line
        years = ['2021', '2022', '2023', '2024']
        progress = [row[f'progress_year{i}'] for i in range(1, 5)]
        
        fig.add_trace(go.Scatter(
            x=years,
            y=progress,
            mode='lines+markers',
            name='Progress',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        
        # Target line
        if 'impact_target' in row:
            fig.add_trace(go.Scatter(
                x=years,
                y=[row['impact_target']] * len(years),
                mode='lines',
                name='Target',
                line=dict(color='gray', width=1, dash='dash')
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
    
    def render_sdg_visualization(self, df: pd.DataFrame):
        """Render SDG treemap visualization"""
        sdg_data = df.groupby(['sdg', 'impact_theme'])['capital_deployed'].sum().reset_index()
        
        # SDG colors from official SDG branding guidelines
        sdg_colors = {
            'SDG1': '#E5243B', 'SDG2': '#DDA63A', 'SDG3': '#4C9F38',
            'SDG4': '#C5192D', 'SDG5': '#FF3A21', 'SDG6': '#26BDE2',
            'SDG7': '#FCC30B', 'SDG8': '#A21942', 'SDG9': '#FD6925',
            'SDG10': '#DD1367', 'SDG11': '#FD9D24', 'SDG12': '#BF8B2E',
            'SDG13': '#3F7E44', 'SDG14': '#0A97D9', 'SDG15': '#56C02B',
            'SDG16': '#00689D', 'SDG17': '#19486A'
        }
        
        fig = px.treemap(
            sdg_data,
            path=['sdg', 'impact_theme'],
            values='capital_deployed',
            title='SDG and Impact Theme Breakdown',
            color='sdg',
            color_discrete_map=sdg_colors
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def render_theme_visualization(self, df: pd.DataFrame):
        """Render theme and outcome visualization"""
        theme_data = df.groupby(['impact_theme', 'impact_outcome']).size().reset_index(name='count')
        
        # Theme colors
        theme_colors = {
            'Climate Change': '#1f77b4',
            'Financial Inclusion': '#2ca02c',
            'Agriculture': '#ff7f0e',
            'Education': '#9467bd',
            'Healthcare': '#d62728'
        }
        
        fig = px.sunburst(
            theme_data,
            path=['impact_theme', 'impact_outcome'],
            values='count',
            title='Impact Themes and Outcomes',
            color='impact_theme',
            color_discrete_map=theme_colors
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate portfolio-level metrics"""
        df = st.session_state.data
        
        # Calculate fund-level metrics
        fund_metrics = df.groupby('fund').agg({
            'company': lambda x: len(set(x)) >= 3,
            'impact_baseline': lambda x: x.notna().mean() >= 0.75,
            'impact_target': lambda x: x.notna().mean() >= 0.75,
            'relevance': lambda x: (x == 'High').mean() >= 0.75
        })
        
        total_funds = len(fund_metrics)
        metrics = {
            'tracking_coverage': (fund_metrics['company'].sum() / total_funds) * 100,
            'baseline_coverage': (fund_metrics['impact_baseline'].sum() / total_funds) * 100,
            'portfolio_targets': (fund_metrics['impact_target'].sum() / total_funds) * 100
        }
        
        # Company targets
        company_targets = df.groupby(['fund', 'company'])['impact_target'].apply(
            lambda x: x.notna().any()
        ).groupby('fund').mean()
        metrics['company_targets'] = (company_targets >= 0.75).mean() * 100
        
        # High relevance metrics
        metrics['high_relevance'] = (fund_metrics['relevance'].sum() / total_funds) * 100
        
        # YoY progress
        progress_cols = ['progress_year1', 'progress_year2', 'progress_year3', 'progress_year4']
        yoy_tracking = df.groupby('fund')[progress_cols].apply(
            lambda x: x.notna().all(axis=1).mean() >= 0.75
        )
        metrics['yoy_progress'] = (yoy_tracking.sum() / total_funds) * 100
        
        return metrics
    
    def render_portfolio_metrics(self):
        """Render portfolio-level metrics"""
        metrics = self.calculate_portfolio_metrics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Fund Coverage",
                f"{metrics['tracking_coverage']:.0f}%",
                help="% funds tracking 75%+ companies"
            )
            st.metric(
                "Baseline Data",
                f"{metrics['baseline_coverage']:.0f}%",
                help="% funds with baseline data collected"
            )
            
        with col2:
            st.metric(
                "Portfolio Targets",
                f"{metrics['portfolio_targets']:.0f}%",
                help="% funds that have established portfolio targets for all metrics"
            )
            st.metric(
                "Company Targets",
                f"{metrics['company_targets']:.0f}%",
                help="% funds that have established company-level impact targets for 75%+ metrics"
            )
            
        with col3:
            st.metric(
                "High Relevance",
                f"{metrics['high_relevance']:.0f}%",
                help="% funds with 75%+ metrics rated “High” for Relevance"
            )
            st.metric(
                "YoY Progress",
                f"{metrics['yoy_progress']:.0f}%",
                help="% funds tracking 2+ years data for 75%+ portfolio companies"
            )

    def render_portfolio_view(self):
        """Render portfolio-level view with expanded columns and clickable funds"""
        # Filter data based on sidebar selections
        df = self.apply_filters(st.session_state.data)
        
        # Create impact results table with headers
        st.subheader("Impact Results")
        
        # Table headers
        cols = st.columns([2, 2, 2, 1.5, 1.5, 3])
        headers = ["Impact Theme", "Impact Outcome", "Impact Metric", "Funds", "Capital Deployed", "Progress"]
        
        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")
        
        # Group by theme, outcome, and metric
        grouped = df.groupby(['impact_theme', 'impact_outcome', 'impact_kpi']).agg({
            'fund': lambda x: sorted(set(x)),
            'capital_deployed': 'sum',
            'progress_year1': 'mean',
            'progress_year2': 'mean',
            'progress_year3': 'mean',
            'progress_year4': 'mean',
            'impact_target': 'mean'
        }).reset_index()
        
        # Display portfolio rows with expand/collapse functionality
        for _, row in grouped.iterrows():
            cols = st.columns([2, 2, 2, 1.5, 1.5, 3])
            
            # Create unique row identifier
            row_id = f"{row['impact_theme']}_{row['impact_outcome']}_{row['impact_kpi']}"
            
            with cols[0]:
                st.write(row['impact_theme'])
            with cols[1]:
                st.write(row['impact_outcome'])
            with cols[2]:
                st.write(row['impact_kpi'])
            with cols[3]:
                funds_count = len(row['fund'])
                funds_str = ", ".join(row['fund']) if funds_count <= 3 else f"{funds_count} funds"
                
                # Make the funds clickable
                if st.button(funds_str, key=f"funds_{row_id}"):
                    if row_id in st.session_state.expanded_rows:
                        st.session_state.expanded_rows.remove(row_id)
                    else:
                        st.session_state.expanded_rows.add(row_id)
            
            with cols[4]:
                st.write(f"${row['capital_deployed']/1000000:.2f}M")
            with cols[5]:
                progress_data = {
                    'impact_target': row['impact_target'],
                    'progress_year1': row['progress_year1'],
                    'progress_year2': row['progress_year2'],
                    'progress_year3': row['progress_year3'],
                    'progress_year4': row['progress_year4']
                }
                fig = self.create_progress_chart(progress_data)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show fund breakdown if expanded
            if row_id in st.session_state.expanded_rows:
                self.render_fund_view(df, row)
                
        # Add SDG and theme visualizations
        col1, col2 = st.columns(2)
        with col1:
            self.render_sdg_visualization(df)
        with col2:
            self.render_theme_visualization(df)

    def render_fund_view(self, df: pd.DataFrame, portfolio_row: pd.Series):
        """Render fund-level breakdown for an expanded portfolio row"""
        fund_data = df[
            (df['impact_theme'] == portfolio_row['impact_theme']) &
            (df['impact_outcome'] == portfolio_row['impact_outcome']) &
            (df['impact_kpi'] == portfolio_row['impact_kpi'])
        ]
        
        for fund in sorted(fund_data['fund'].unique()):
            fund_rows = fund_data[fund_data['fund'] == fund]
            
            fund_metrics = {
                'capital_deployed': fund_rows['capital_deployed'].sum(),
                'progress_year1': fund_rows['progress_year1'].mean(),
                'progress_year2': fund_rows['progress_year2'].mean(),
                'progress_year3': fund_rows['progress_year3'].mean(),
                'progress_year4': fund_rows['progress_year4'].mean(),
                'impact_target': fund_rows['impact_target'].mean()
            }
            
            cols = st.columns([2, 2, 2, 1.5, 1.5, 3])
            
            # Create unique fund row ID
            fund_row_id = f"{portfolio_row['impact_theme']}_{portfolio_row['impact_outcome']}_{portfolio_row['impact_kpi']}_{fund}"
            
            with cols[0]:
                st.write("  ")  # Indent
            with cols[1]:
                st.write(fund)  # Show fund name
            with cols[2]:
                st.write("  ")  # Keep same metric
            with cols[3]:
                # Get list of companies and make it clickable
                companies = fund_rows['company'].unique()
                company_count = len(companies)
                company_text = f"{company_count} companies"
                
                # Make companies clickable as a button
                if st.button(company_text, key=f"companies_{fund_row_id}"):
                    if fund_row_id in st.session_state.expanded_rows:
                        st.session_state.expanded_rows.remove(fund_row_id)
                    else:
                        st.session_state.expanded_rows.add(fund_row_id)
                        
            with cols[4]:
                st.write(f"${fund_metrics['capital_deployed']/1000000:.2f}M")
            with cols[5]:
                fig = self.create_progress_chart(fund_metrics)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show company breakdown if expanded
            if fund_row_id in st.session_state.expanded_rows:
                for _, company_row in fund_rows.iterrows():
                    cols = st.columns([2, 2, 2, 1.5, 1.5, 3])
                    
                    with cols[0]:
                        st.write("  ")  # Double indent
                    with cols[1]:
                        st.write("  ")  # Space
                    with cols[2]:
                        st.write(company_row['company'])  # Show company name
                    with cols[3]:
                        st.write("  ")  # No count needed
                    with cols[4]:
                        st.write(f"${company_row['capital_deployed']/1000000:.2f}M")
                    with cols[5]:
                        company_data = {
                            'impact_target': company_row['impact_target'],
                            'progress_year1': company_row['progress_year1'],
                            'progress_year2': company_row['progress_year2'],
                            'progress_year3': company_row['progress_year3'],
                            'progress_year4': company_row['progress_year4'] if 'progress_year4' in company_row else company_row['progress_year3'] * 1.1
                        }
                        fig = self.create_progress_chart(company_data)
                        st.plotly_chart(fig, use_container_width=True)

    def render_company_view(self):
        """Render company-level view"""
        if not st.session_state.selected_fund:
            st.error("No fund selected")
            return
            
        st.subheader(f"Company Performance: {st.session_state.selected_fund}")
        
        # Back button
        if st.button("← Back to Fund View"):
            st.session_state.drill_level = 'fund'
            st.session_state.selected_fund = None
            st.rerun()
            
        # Filter data
        df = self.apply_filters(st.session_state.data)
        df = df[df['fund'] == st.session_state.selected_fund]
        
        # Table headers
        cols = st.columns([2, 2, 2, 1.5, 1.5, 3])
        headers = ["Company", "Impact Outcome", "Impact Metric", "Baseline", "Capital Deployed", "Progress"]
        
        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")
        
        # Display company results
        for _, row in df.iterrows():
            cols = st.columns([2, 2, 2, 1.5, 1.5, 3])
            
            with cols[0]:
                st.write(row['company'])
            with cols[1]:
                st.write(row['impact_outcome'])
            with cols[2]:
                st.write(row['impact_kpi'])
            with cols[3]:
                st.write(f"{row['impact_baseline']:.0f}")
            with cols[4]:
                st.write(f"${row['capital_deployed']/1000000:.2f}M")
            with cols[5]:
                progress_data = {
                    'impact_target': row['impact_target'],
                    'progress_year1': row['progress_year1'],
                    'progress_year2': row['progress_year2'],
                    'progress_year3': row['progress_year3'],
                    'progress_year4': row['progress_year4'] if 'progress_year4' in row else row['progress_year3'] * 1.1
                }
                fig = self.create_progress_chart(progress_data)
                st.plotly_chart(fig, use_container_width=True)

    def render_dashboard(self):
        """Render main dashboard"""
        st.title("Impact Dashboard")
        
        if st.session_state.data.empty:
            st.warning("No data available. Please import data first.")
            return
            
        # Render filters at the top
        self.render_dashboard_filters()
        
        # Portfolio metrics
        self.render_portfolio_metrics()
        
        # Main content based on drill level
        if st.session_state.drill_level == 'portfolio':
            self.render_portfolio_view()
        elif st.session_state.drill_level == 'fund':
            self.render_fund_view()
        else:
            self.render_company_view()

    def get_template_file(self) -> bytes:
        """Generate template file"""
        df = pd.DataFrame({
            'fund': ['Example Fund'],
            'company': ['Example Company'],
            'impact_theme': ['Climate Change'],
            'impact_outcome': ['Reduced carbon emissions'],
            'geography': ['Global'],
            'capital_deployed': [1000000],
            'sdg': ['SDG13'],
            'impact_kpi': ['CO2 Emissions Avoided (tonnes)'],
            'impact_baseline': [100],
            'impact_target': [500],
            'progress_year1': [200],
            'progress_year2': [300],
            'progress_year3': [400],
            'progress_year4': [450],
            'relevance': ['High'],
            'has_narrative': [False],
            'narrative_type': [None],
            'global_benchmark': [75]
        })
        
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        return buffer.getvalue()

    def render_upload_options(self):
        """Render initial upload options"""
        st.subheader("Choose Import Method")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Upload Existing Data")
            uploaded_file = st.file_uploader(
                "Drag and drop file",
                type=['csv', 'xlsx'],
                help="Upload your impact data file"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                        
                    st.session_state.uploaded_data = df
                    st.success("File uploaded successfully!")
                    
                    if st.button("Continue →"):
                        st.session_state.upload_stage = 1
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        with col2:
            st.markdown("### Download Template")
            if st.download_button(
                "Download Template",
                self.get_template_file(),
                "impact_data_template.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ):
                st.success("Template downloaded!")

    def render_fund_selection(self):
        """Render fund and company selection interface"""
        st.subheader("Select Funds and Companies")
        
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            
            # Get unique funds and companies
            funds = df['fund'].unique()
            
            # Initialize selection state if needed
            if 'selected_funds' not in st.session_state:
                st.session_state.selected_funds = list(funds)
                
            if 'selected_companies' not in st.session_state:
                st.session_state.selected_companies = list(df['company'].unique())
            
            # Create expandable sections for each fund
            for fund in funds:
                with st.expander(f"{fund}", expanded=True):
                    # Fund checkbox
                    fund_selected = st.checkbox(
                        f"Select all companies in {fund}",
                        value=fund in st.session_state.selected_funds,
                        key=f"fund_{fund}"
                    )
                    
                    # Update selected funds
                    if fund_selected and fund not in st.session_state.selected_funds:
                        st.session_state.selected_funds.append(fund)
                    elif not fund_selected and fund in st.session_state.selected_funds:
                        st.session_state.selected_funds.remove(fund)
                    
                    # Company checkboxes
                    companies = df[df['fund'] == fund]['company'].unique()
                    for company in companies:
                        company_selected = st.checkbox(
                            company,
                            value=company in st.session_state.selected_companies,
                            key=f"company_{fund}_{company}"
                        )
                        
                        if company_selected and company not in st.session_state.selected_companies:
                            st.session_state.selected_companies.append(company)
                        elif not company_selected and company in st.session_state.selected_companies:
                            st.session_state.selected_companies.remove(company)
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back"):
                    st.session_state.upload_stage = 0
                    st.rerun()
                    
            with col2:
                if st.button("Continue →"):
                    # Filter data based on selection
                    filtered_df = df[
                        (df['fund'].isin(st.session_state.selected_funds)) &
                        (df['company'].isin(st.session_state.selected_companies))
                    ]
                    
                    st.session_state.uploaded_data = filtered_df
                    self.calculate_data_completeness()
                    st.session_state.upload_stage = 2
                    st.rerun()

    def calculate_data_completeness(self):
        """Calculate completeness for each dimension"""
        df = st.session_state.uploaded_data
        
        dimensions = {
            'impact_theme': 'Impact theme',
            'impact_outcome': 'Impact outcome',
            'geography': 'Geography',
            'capital_deployed': 'Capital invested',
            'sdg': 'SDG',
            'impact_kpi': 'Impact KPI',
            'impact_baseline': 'Impact baseline',
            'impact_target': 'Impact target'
        }
        
        # Check for YoY progress columns
        progress_cols = [col for col in df.columns 
                        if re.search(r'progress.*year\d+|year\d+.*progress', col.lower())]
        
        if progress_cols:
            dimensions['yoy_progress'] = 'YoY Progress for the past 3 years'
        
        # Calculate completeness for each dimension
        completeness_status = {}
        for key, label in dimensions.items():
            if key == 'yoy_progress':
                missing_count = df[progress_cols].isna().sum().sum()
                total_cells = len(df) * len(progress_cols)
                
                if missing_count == total_cells:
                    status = "Missing"
                elif missing_count > 0:
                    status = "Partially Complete"
                else:
                    status = "Fully Complete"
            else:
                if key not in df.columns:
                    status = "Missing"
                else:
                    missing_count = df[key].isna().sum()
                    if missing_count == len(df):
                        status = "Missing"
                    elif missing_count > 0:
                        status = "Partially Complete"
                    else:
                        status = "Fully Complete"
                        
            completeness_status[label] = status
            
        st.session_state.dimensions_status = completeness_status

    def render_validation_report(self):
        """Render data validation report card"""
        st.subheader("Data Validation Report")
        
        if not st.session_state.dimensions_status:
            st.error("No validation data available")
            return
            
        # Display status with colored text
        for dimension, status in st.session_state.dimensions_status.items():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"**{dimension}**")
                
            with col2:
                color = {
                    "Missing": "red",
                    "Partially Complete": "orange",
                    "Fully Complete": "green"
                }[status]
                st.markdown(f'<span style="color:{color}">{status}</span>', 
                          unsafe_allow_html=True)
            
            with col3:
                if status in ["Missing", "Partially Complete"]:
                    if st.button("Edit", key=f"edit_{dimension}"):
                        st.session_state.dimension_to_edit = dimension
                        st.session_state.upload_stage = 3
                        st.rerun()
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.upload_stage = 1
                st.rerun()
                
        with col2:
            if st.button("Complete Import →"):
                st.session_state.data = st.session_state.uploaded_data
                st.session_state.page = 'dashboard'
                st.rerun()

    def render_data_editor(self):
        """Render data editor interface"""
        if 'dimension_to_edit' not in st.session_state:
            st.error("No dimension selected for editing")
            if st.button("Back"):
                st.session_state.upload_stage = 2
                st.rerun()
            return
            
        dimension = st.session_state.dimension_to_edit
        st.subheader(f"Edit {dimension} Data")
        
        # Get column name from dimension label
        dimension_map = {
            'Impact theme': 'impact_theme',
            'Impact outcome': 'impact_outcome',
            'Geography': 'geography',
            'Capital invested': 'capital_deployed',
            'SDG': 'sdg',
            'Impact KPI': 'impact_kpi',
            'Impact baseline': 'impact_baseline',
            'Impact target': 'impact_target',
            'YoY Progress for the past 3 years': 'yoy_progress'
        }
        
        if dimension == 'YoY Progress for the past 3 years':
            progress_cols = ['progress_year1', 'progress_year2', 'progress_year3', 'progress_year4']
            editable_cols = ['fund', 'company'] + progress_cols
        else:
            col_name = dimension_map.get(dimension)
            if not col_name:
                st.error(f"Cannot find column for {dimension}")
                return
                
            # Add column if it doesn't exist
            if col_name not in st.session_state.uploaded_data.columns:
                st.session_state.uploaded_data[col_name] = np.nan
                
            editable_cols = ['fund', 'company', col_name]
            
        # Create editable dataframe
        edited_df = st.data_editor(
            st.session_state.uploaded_data[editable_cols],
            use_container_width=True
        )
        
        # Save changes
        if st.button("Save Changes"):
            for idx, row in edited_df.iterrows():
                for col in editable_cols:
                    if col not in ['fund', 'company']:
                        st.session_state.uploaded_data.loc[idx, col] = row[col]
                        
            self.calculate_data_completeness()
            st.session_state.upload_stage = 2
            st.rerun()
            
        if st.button("Cancel"):
            st.session_state.upload_stage = 2
            st.rerun()

    def render_import_flow(self):
        """Render data import workflow"""
        st.title("Import Impact Data")
        
        # Show different stages based on upload_stage
        if st.session_state.upload_stage == 0:
            self.render_upload_options()
        elif st.session_state.upload_stage == 1:
            self.render_fund_selection()
        elif st.session_state.upload_stage == 2:
            self.render_validation_report()
        elif st.session_state.upload_stage == 3:
            self.render_data_editor()

    def render_metrics_analysis(self, theme: str, outcome: str):
        """Render metrics analysis section"""
        st.subheader("Common Metrics Analysis")
        
        # Filter data for theme/outcome
        df = st.session_state.data[
            (st.session_state.data['impact_theme'] == theme) &
            (st.session_state.data['impact_outcome'] == outcome)
        ]
        
        # Get metrics
        metrics = df['impact_kpi'].unique()
        
        # Create metrics table with headers
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write("**Impact Metric**")
        with col2:
            st.write("**Coverage**")
        with col3:
            st.write("**Relevance**")
            
        # Create comparison table
        for metric in metrics:
            metric_data = df[df['impact_kpi'] == metric]
            coverage = len(metric_data) / len(df['company'].unique()) * 100
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(metric)
            with col2:
                st.write(f"{coverage:.0f}% coverage")
            with col3:
                relevance = metric_data['relevance'].iloc[0]
                color = {
                    'High': 'green',
                    'Medium': 'orange',
                    'Low': 'red'
                }.get(relevance, 'gray')
                st.markdown(
                    f'<span style="color:{color}">{relevance}</span>',
                    unsafe_allow_html=True
                )

    def render_benchmark_analysis(self, theme: str, outcome: str):
        """Render benchmark analysis section"""
        st.subheader("Performance vs Benchmarks")
        
        # Filter data
        df = st.session_state.data[
            (st.session_state.data['impact_theme'] == theme) &
            (st.session_state.data['impact_outcome'] == outcome)
        ]
        
        # Calculate performance
        current_performance = (df['progress_year4'] / df['impact_target']).mean() * 100
        benchmark = df['global_benchmark'].mean() if 'global_benchmark' in df.columns else 75
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+delta",
            value=current_performance,
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
            title={'text': "Portfolio Performance vs Benchmark"}
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Company performance table headers
        st.subheader("Company Performance")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("**Company**")
        with col2:
            st.write("**Progress**")
            
        # Company performance rows
        for _, company_data in df.groupby('company'):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(company_data['company'].iloc[0])
            with col2:
                progress_data = {
                    'impact_target': company_data['impact_target'].iloc[0],
                    'progress_year1': company_data['progress_year1'].iloc[0],
                    'progress_year2': company_data['progress_year2'].iloc[0],
                    'progress_year3': company_data['progress_year3'].iloc[0],
                    'progress_year4': company_data['progress_year4'].iloc[0] if 'progress_year4' in company_data.columns else company_data['progress_year3'].iloc[0] * 1.1
                }
                fig = self.create_progress_chart(progress_data)
                st.plotly_chart(fig, use_container_width=True)

    def render_impact_analysis(self):
        """Render impact analysis page"""
        st.title("Impact Analysis")
        
        if st.session_state.data.empty:
            st.warning("No data available. Please import data first.")
            return
            
        # Theme and outcome selection
        col1, col2 = st.columns(2)
        with col1:
            selected_theme = st.selectbox(
                "Select Impact Theme",
                options=sorted(st.session_state.data['impact_theme'].unique())
            )
        with col2:
            selected_outcome = st.selectbox(
                "Select Impact Outcome",
                options=sorted(st.session_state.data[
                    st.session_state.data['impact_theme'] == selected_theme
                ]['impact_outcome'].unique())
            )
            
        # Analysis sections
        self.render_metrics_analysis(selected_theme, selected_outcome)
        self.render_benchmark_analysis(selected_theme, selected_outcome)

    def render_narrative_upload(self):
        """Render narrative upload section"""
        st.subheader("Add Impact Story")
        
        # Company selection
        company = st.selectbox(
            "Select Company",
            options=sorted(st.session_state.data['company'].unique())
        )
        
        # Upload controls
        col1, col2, col3 = st.columns(3)
        uploaded_type = None
        
        with col1:
            video = st.file_uploader("Upload Video", type=['mp4', 'mov'])
            if video:
                uploaded_type = 'video'
                
        with col2:
            text = st.text_area("Written Story")
            if text:
                uploaded_type = 'text'
                
        with col3:
            image = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg'])
            if image:
                uploaded_type = 'image'
        
        if uploaded_type and st.button("Save Story"):
            # Update dataframe
            mask = st.session_state.data['company'] == company
            st.session_state.data.loc[mask, 'has_narrative'] = True
            st.session_state.data.loc[mask, 'narrative_type'] = uploaded_type
            st.success(f"Story saved for {company}")

    def render_existing_stories(self):
        """Render existing impact stories"""
        st.subheader("Existing Stories")
        
        if 'has_narrative' not in st.session_state.data.columns:
            st.session_state.data['has_narrative'] = False
            st.session_state.data['narrative_type'] = None
            st.warning("No stories available yet. Please add some stories.")
            return
            
        stories = st.session_state.data[st.session_state.data['has_narrative'] == True].copy()
        
        if len(stories) == 0:
            st.warning("No stories available yet. Please add some stories.")
            return
            
        for _, story in stories.iterrows():
            with st.expander(f"{story['impact_theme']} - {story['company']}"):
                if story['narrative_type'] == 'video':
                    st.write("Video story available")
                elif story['narrative_type'] == 'text':
                    st.write("Text story available")
                elif story['narrative_type'] == 'image':
                    st.write("Image story available")

    def render_missing_coverage(self):
        """Render missing coverage analysis"""
        st.subheader("Missing Impact Coverage")
        
        # Get themes/outcomes without stories or data
        df = st.session_state.data
        all_themes = df['impact_theme'].unique()
        all_outcomes = df['impact_outcome'].unique()
        
        if 'has_narrative' not in df.columns:
            df['has_narrative'] = False
        
        missing = []
        for theme in all_themes:
            theme_data = df[df['impact_theme'] == theme]
            for outcome in all_outcomes:
                outcome_data = theme_data[theme_data['impact_outcome'] == outcome]
                
                if len(outcome_data) == 0 or not outcome_data['has_narrative'].any():
                    missing.append({
                        'theme': theme,
                        'outcome': outcome,
                        'missing': 'Both' if len(outcome_data) == 0 else 'Story'
                    })
        
        if missing:
            st.write("The following impact areas are missing data or stories:")
            for item in missing:
                st.markdown(
                    f"**{item['theme']} - {item['outcome']}**: Missing {item['missing']}"
                )
        else:
            st.success("All themes and outcomes have data and stories!")

    def render_impact_narrative(self):
        """Render impact narrative page"""
        st.title("Impact Narrative")
        
        tab1, tab2 = st.tabs(["Stories", "Missing Coverage"])
        
        with tab1:
            self.render_narrative_upload()
            self.render_existing_stories()
            
        with tab2:
            self.render_missing_coverage()
            
    def run(self):
        """Main dashboard function"""
        pages = {
            'dashboard': self.render_dashboard,
            'import_data': self.render_import_flow,
            'impact_analysis': self.render_impact_analysis,
            'impact_narrative': self.render_impact_narrative
        }
        
        if st.session_state.page in pages:
            pages[st.session_state.page]()
        else:
            st.error("Page not found")


if __name__ == "__main__":
    dashboard = ImpactDashboard()
    dashboard.run()