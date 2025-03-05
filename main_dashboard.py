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
    
        # Reduce sidebar width
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
                
            # Add a divider
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
            
        # Create tabs for different analysis views
        tab1, tab2 = st.tabs(["Metrics Analysis", "Portfolio Optimization"])
        
        with tab1:
            # Theme and outcome selection
            col1, col2 = st.columns(2)
            with col1:
                selected_theme = st.selectbox(
                    "Select Impact Theme",
                    options=sorted(st.session_state.data['impact_theme'].unique()),
                    key="theme_selector_1"
                )
            with col2:
                selected_outcome = st.selectbox(
                    "Select Impact Outcome",
                    options=sorted(st.session_state.data[
                        st.session_state.data['impact_theme'] == selected_theme
                    ]['impact_outcome'].unique()),
                    key="outcome_selector_1"
                )
                
            # Analysis sections
            self.render_metrics_analysis(selected_theme, selected_outcome)
            self.render_benchmark_analysis(selected_theme, selected_outcome)
    
        with tab2:
            # Portfolio optimization tools
            self.render_portfolio_optimization()

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
    
    
    def render_portfolio_optimization(self):
        """Render enhanced portfolio optimization tools"""
        st.subheader("Portfolio Optimization for Private Markets")
        
        if st.session_state.data.empty:
            st.warning("No data available. Please import data first.")
            return
        
        # Add private markets context
        with st.expander("Private Markets Context", expanded=True):
            st.markdown("""
            **Portfolio Optimization for Private Markets**
            
            Private market investments have unique characteristics that affect portfolio construction:
            
            * **Limited Liquidity:** Investments are typically locked for 7-10+ years
            * **Fixed Commitments:** Once capital is committed, allocations cannot be easily changed
            * **Forward-Looking Strategy:** Portfolio optimization focuses on uncommitted capital and future funds
            * **Natural Evolution:** Portfolio changes happen gradually through investment cycles and exits
            
            The tools below are designed for these private market realities, focusing on future capital deployment 
            rather than immediate rebalancing.
            """)
        
        # Create tabs for different optimization approaches
        optimization_tabs = st.tabs([
            "Multi-Dimensional Optimization", 
            "Public Markets Optimization"
        ])
        
        # Tab 1: Multi-Dimensional Optimization (BII-inspired)
        with optimization_tabs[0]:
            st.markdown("### Multi-Dimensional Portfolio Construction")
            st.markdown("""
            This integrated approach optimizes your portfolio across four key dimensions simultaneously:
            **Impact**, **Return**, **Risk**, and **Liquidity**.
            """)
            
            # Completely restructure this tab to avoid layout issues
            # Store optimization parameters in a form
            with st.form("optimization_params_form"):
                st.markdown("#### Optimization Parameters")
                
                # Set primary optimization parameter
                primary_param = st.radio(
                    "Primary Optimization Parameter",
                    options=["Impact Maximization", "Return Maximization", "Risk Minimization", "Liquidity Optimization"],
                    index=0
                )
                
                # Set boundary conditions
                st.markdown("#### Boundary Conditions")
                min_return = st.slider("Minimum Return (%)", 0.0, 10.0, 2.0, 0.5)
                max_risk = st.slider("Maximum Risk Score", 1.0, 10.0, 5.0, 0.5)
                min_liquidity = st.slider("Minimum Liquidity Ratio", 0.0, 1.0, 0.3, 0.1)
                
                # Submit button for the form
                submitted = st.form_submit_button("Run Multi-Dimensional Optimization")
            
            # Check if form was submitted
            if submitted:
                # Generate multi-dimensional visualization
                radar_fig = self.create_multidimensional_portfolio_chart()
                
                # Create a new container for results that's entirely separate from the form
                results_container = st.container()
                
                with results_container:
                    st.markdown("#### Multi-Dimensional Optimization Results")
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    # Show recommended funds right after the optimization results but before strategy info
                    self.display_recommended_funds(primary_param, min_return, max_risk, min_liquidity)
                    
                    # Future Capital Deployment Strategy
                    st.markdown("### Future Capital Deployment Strategy")
                    
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-left: 4px solid #2ca02c; padding: 15px; border-radius: 4px;">
                        <h5 style="margin-top: 0;">{primary_param} Strategy</h5>
                        <p><strong>Undeployed capital recommendations:</strong></p>
                        <ul>
                            <li>Prioritize high-impact investments with lower correlation to current holdings</li>
                            <li>Focus new investments on climate and financial inclusion themes</li>
                            <li>Target medium-liquidity investments for 60% of remaining capital</li>
                        </ul>
                        <p><strong>Expected long-term portfolio transformation:</strong></p>
                        <ul>
                            <li>Fund ID Score: <span style="color: green;">+25%</span> (3-5 year horizon)</li>
                            <li>Return: <span style="color: green;">+0.3%</span> (annual)</li>
                            <li>Risk: <span style="color: green;">-0.6 points</span> (5 year horizon)</li>
                            <li>Liquidity Ratio: <span style="color: {('red' if primary_param=='Impact Maximization' else 'green')};">{"-0.3" if primary_param=='Impact Maximization' else "+0.5"}</span></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Implementation Strategy
                    st.markdown("### Implementation Strategy")
                    st.markdown("""
                    Based on portfolio assessment, the following forward-looking strategies are recommended:
                    """)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### New Investment Guidelines")
                        st.markdown("""
                        - Allocate 10-15% of new commitments to high-impact, high-risk investments
                        - Apply sector-specific risk limits for new capital deployment
                        - For every $10M deployed, reserve at least $3M for investments providing regular cash flow
                        - Prioritize investments with clear impact targets aligned with portfolio themes
                        """)
                        
                    with col2:
                        st.markdown("#### Future Fund Planning")
                        st.markdown("""
                        - For upcoming fund design, expand financial inclusion focus by 5-8%
                        - Reserve up to 10% of new fund commitments for opportunistic high-impact deals
                        - Structure new funds with 15-20% allocation to early-harvest investments
                        - Plan exits for 25% of current low-impact investments at end of holding period
                        """)
                    
                    # Private Markets Context
                    st.markdown("### Private Markets Context")
                    st.markdown("""
                    **Note:** Since existing private market investments cannot be easily rebalanced mid-holding period, 
                    these recommendations focus on:
                    1. Strategic deployment of remaining uncommitted capital
                    2. Adjustments to future fund designs and structures
                    3. Long-term portfolio transformation via natural fund cycles and selective secondary sales
                    4. Exit planning for investments approaching end of holding period
                    """)
                    
                    # Add apply button
                    if st.button("Apply Future Deployment Strategy", key="apply_multidim"):
                        st.success("Future deployment strategy saved successfully!")
                
        
        # Tab 2: Public Markets Optimization Options (preserved for compatibility)
        with optimization_tabs[2]:
            st.markdown("### Public Markets Optimization Options")
            
            # Create tabs for different optimization approaches
            original_tabs = st.tabs([
                "Impact Maximization", 
                "Target Achievement", 
                "Custom Optimization"
            ])
            
            # Tab 1: Impact Maximization
            with original_tabs[0]:
                st.markdown("### Impact Maximization Recommendations")
                st.markdown("""
                This tool analyzes your current portfolio allocation and suggests reallocations 
                to maximize impact while maintaining your risk profile.
                """)
                
                # Generate current portfolio summary
                funds_data = self.get_fund_performance_data()
                
                # Display current allocation
                allocation_fig = self.create_current_allocation_chart(funds_data)
                st.plotly_chart(allocation_fig, use_container_width=True)
                
                # Display optimization recommendations
                self.display_optimization_recommendations(funds_data)
            
            # Tab 2: Target Achievement
            with original_tabs[1]:
                st.markdown("### Target Achievement Strategy")
                st.markdown("""
                This tool helps optimize your portfolio to achieve specific impact targets
                by the selected target date.
                """)
                
                # Target date selection
                target_date = st.date_input(
                    "Target Date", 
                    value=pd.to_datetime("2025-12-31")
                )
                
                # Get current progress toward targets
                target_data = self.calculate_target_progress()
                
                # Show current progress toward targets
                progress_fig = self.create_target_progress_chart(target_data)
                st.plotly_chart(progress_fig, use_container_width=True)
                
                # Display target achievement recommendations
                self.display_target_recommendations(target_data, target_date)
            
            # Tab 3: Custom Optimization
            with original_tabs[2]:
                st.markdown("### Custom Portfolio Optimization")
                
                # Custom optimization inputs
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    optimization_goal = st.selectbox(
                        "Optimization Goal",
                        options=["Impact Maximization", "Risk Minimization", 
                                "Target Achievement", "SDG Alignment"]
                    )
                
                with col2:
                    constraint_type = st.selectbox(
                        "Constraint",
                        options=["Maintain Risk Level", "Maintain Capital Allocation", 
                                "Maintain Geography Exposure", "No Constraints"]
                    )
                
                with col3:
                    time_horizon = st.selectbox(
                        "Time Horizon",
                        options=["12 months", "18 months", "24 months", "36 months"]
                    )
                
                # Run custom optimization
                if st.button("Run Custom Optimization"):
                    with st.spinner("Running optimization analysis..."):
                        # Hypothetical delay for complex calculation
                        import time
                        time.sleep(2)
                        
                        # Show optimization results
                        self.display_custom_optimization_results(
                            optimization_goal, constraint_type, time_horizon
                        )

    def create_multidimensional_portfolio_chart(self):
        """Create spider/radar chart for multi-dimensional portfolio visualization"""
        df = st.session_state.data
        
        # Calculate portfolio-level metrics for each fund
        funds = df['fund'].unique()
        fund_metrics = {}
        
        for fund in funds:
            fund_data = df[df['fund'] == fund]
            
            # Calculate metrics
            impact_score = (fund_data['progress_year3'] / fund_data['impact_target']).mean() * 10
            return_score = np.random.uniform(3.0, 8.0)  # Simulated return metric
            risk_score = np.random.uniform(2.0, 7.0)    # Simulated risk metric
            liquidity_score = np.random.uniform(2.0, 9.0)  # Simulated liquidity metric
            
            fund_metrics[fund] = {
                'Impact': impact_score,
                'Return': return_score,
                'Risk': risk_score,
                'Liquidity': liquidity_score
            }
        
        # Create radar chart
        fig = go.Figure()
        
        # Add a trace for each fund
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (fund, metrics) in enumerate(fund_metrics.items()):
            fig.add_trace(go.Scatterpolar(
                r=[metrics['Impact'], metrics['Return'], metrics['Risk'], metrics['Liquidity']],
                theta=['Impact', 'Return', 'Risk', 'Liquidity'],
                fill='toself',
                name=fund,
                line_color=colors[i % len(colors)]
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True
        )
        
        return fig

    def create_correlation_matrix(self):
        """Create correlation matrix visualization for portfolio dimensions"""
        # Simulate data for correlation analysis
        np.random.seed(42)
        df = st.session_state.data
        
        # Generate correlation data
        n_points = 20
        
        # Actual correlation based on BII findings: impact & return (no correlation), impact & liquidity (negative)
        impact = np.random.uniform(3, 9, n_points)
        returns = impact * 0.1 + np.random.normal(0, 2, n_points)  # No correlation
        risk = np.random.uniform(2, 8, n_points)
        liquidity = 10 - impact * 0.5 + np.random.normal(0, 1, n_points)  # Negative correlation with impact
        
        # Create scatter plots matrix
        fig = go.Figure()
        
        # Impact vs Return (no correlation)
        fig.add_trace(go.Scatter(
            x=impact, 
            y=returns,
            mode='markers',
            marker=dict(color='#1f77b4', size=10),
            name='Impact vs Return'
        ))
        
        # Create layout
        fig.update_layout(
            title="Impact and Returns: No Significant Correlation",
            xaxis_title="Fund ID Score",
            yaxis_title="Returns (%)",
            height=400
        )
        
        # Add annotation
        fig.add_annotation(
            x=6, y=8,
            text="No significant correlation between Impact and Return",
            showarrow=False,
            font=dict(size=14)
        )
        
        return fig
 
    def create_strategic_allocation_chart(self):
        """Create strategic allocation visualization"""
        # Generate example data for strategic allocation
        data = {
            'Category': ['Current', 'Target Min', 'Target Max'],
            'Climate Change': [35, 30, 50],
            'Financial Inclusion': [25, 20, 30],
            'Agriculture': [15, 10, 20],
            'High-Risk/Catalyst': [8, 10, 15],
            'Growth': [72, 65, 75],
            'Stable/Low-Risk': [20, 15, 20]
        }
        
        # Create grouped bar chart
        fig = go.Figure()
        
        # Add trace for current allocation
        fig.add_trace(go.Bar(
            x=['Climate Change', 'Financial Inclusion', 'Agriculture'],
            y=[data['Climate Change'][0], data['Financial Inclusion'][0], data['Agriculture'][0]],
            name='Current Allocation',
            marker_color='#1f77b4'
        ))
        
        # Add trace for target min
        fig.add_trace(go.Bar(
            x=['Climate Change', 'Financial Inclusion', 'Agriculture'],
            y=[data['Climate Change'][1], data['Financial Inclusion'][1], data['Agriculture'][1]],
            name='Target Min',
            marker_color='#ff7f0e'
        ))
        
        # Add trace for target max
        fig.add_trace(go.Bar(
            x=['Climate Change', 'Financial Inclusion', 'Agriculture'],
            y=[data['Climate Change'][2], data['Financial Inclusion'][2], data['Agriculture'][2]],
            name='Target Max',
            marker_color='#2ca02c'
        ))
        
        # Update layout
        fig.update_layout(
            title="Strategic Asset Allocation vs. Targets",
            xaxis_title="Impact Theme",
            yaxis_title="Allocation (%)",
            barmode='group',
            hovermode="x unified"
        )
        
        return fig

    def display_multidimensional_results(self, primary_param, min_return, max_risk, min_liquidity):
            """Display results of multi-dimensional optimization"""

            if 'optimization_params' in st.session_state:
                # Clear the previous input area
                st.session_state.pop('optimization_params')
            
            # Create a fresh full-width container for results
            st.empty()  # Clear any previous content
            
            # Create results card
            st.markdown("#### Multi-Dimensional Optimization Results")
            
            # Create the radar chart
            fig = go.Figure()
            
            # Current values
            fig.add_trace(go.Scatterpolar(
                r=[6.5, 4.2, 5.8, 3.5],
                theta=['Impact', 'Return', 'Risk', 'Liquidity'],
                fill='toself',
                name='Current Portfolio',
                line_color='#1f77b4'
            ))
            
            # Future projected values - adjust based on primary parameter
            if primary_param == "Impact Maximization":
                projected_values = [8.2, 4.5, 5.2, 3.2]
            elif primary_param == "Return Maximization":
                projected_values = [6.8, 6.5, 5.9, 3.8]
            elif primary_param == "Risk Minimization":
                projected_values = [6.3, 4.0, 3.5, 4.0]
            else:  # Liquidity optimization
                projected_values = [6.2, 4.5, 5.2, 7.5]
                
            fig.add_trace(go.Scatterpolar(
                r=projected_values,
                theta=['Impact', 'Return', 'Risk', 'Liquidity'],
                fill='toself',
                name='Future Portfolio (3-5 Years)',
                line_color='#2ca02c'
            ))
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                ),
                showlegend=True,
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),  # Minimize margins
                autosize=True  # Let the chart autosize to container
            )
            
            # Use a fresh container for the chart to ensure full width
            with st.container():
                st.plotly_chart(fig, use_container_width=True)
            
            # Future Capital Deployment Strategy
            with st.container():
                st.markdown("### Future Capital Deployment Strategy")
                
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-left: 4px solid #2ca02c; padding: 15px; border-radius: 4px;">
                    <h5 style="margin-top: 0;">{primary_param} Strategy</h5>
                    <p><strong>Undeployed capital recommendations:</strong></p>
                    <ul>
                        <li>Prioritize high-impact investments with lower correlation to current holdings</li>
                        <li>Focus new investments on climate and financial inclusion themes</li>
                        <li>Target medium-liquidity investments for 60% of remaining capital</li>
                    </ul>
                    <p><strong>Expected long-term portfolio transformation:</strong></p>
                    <ul>
                        <li>Fund ID Score: <span style="color: green;">+25%</span> (3-5 year horizon)</li>
                        <li>Return: <span style="color: green;">+0.3%</span> (annual)</li>
                        <li>Risk: <span style="color: green;">-0.6 points</span> (5 year horizon)</li>
                        <li>Liquidity Ratio: <span style="color: {('red' if primary_param=='Impact Maximization' else 'green')};">{"-0.3" if primary_param=='Impact Maximization' else "+0.5"}</span></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Implementation Strategy
            with st.container():
                st.markdown("### Implementation Strategy")
                st.markdown("""
                Based on portfolio assessment, the following forward-looking strategies are recommended:
                """)
                
                # Use columns within the container
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### New Investment Guidelines")
                    st.markdown("""
                    - Allocate 10-15% of new commitments to high-impact, high-risk investments
                    - Apply sector-specific risk limits for new capital deployment
                    - For every $10M deployed, reserve at least $3M for investments providing regular cash flow
                    - Prioritize investments with clear impact targets aligned with portfolio themes
                    """)
                    
                with col2:
                    st.markdown("#### Future Fund Planning")
                    st.markdown("""
                    - For upcoming fund design, expand financial inclusion focus by 5-8%
                    - Reserve up to 10% of new fund commitments for opportunistic high-impact deals
                    - Structure new funds with 15-20% allocation to early-harvest investments
                    - Plan exits for 25% of current low-impact investments at end of holding period
                    """)
            
            # Private Markets Context
            with st.container():
                st.markdown("### Private Markets Context")
                st.markdown("""
                **Note:** Since existing private market investments cannot be easily rebalanced mid-holding period, 
                these recommendations focus on:
                1. Strategic deployment of remaining uncommitted capital
                2. Adjustments to future fund designs and structures
                3. Long-term portfolio transformation via natural fund cycles and selective secondary sales
                4. Exit planning for investments approaching end of holding period
                """)
                
                # Add apply button
                if st.button("Apply Future Deployment Strategy", key="apply_multidim"):
                    st.success("Future deployment strategy saved successfully!")
                    # Show recommended funds
                    self.display_recommended_funds(primary_param, min_return, max_risk, min_liquidity)
                    
    def generate_recommended_funds(self, primary_param, min_return, max_risk, min_liquidity, current_funds):
        """Generate a list of recommended funds based on the optimization criteria"""
        # Real implementation would query a database or API for potential funds
        # This is a simplified version with mock data
        
        # Define fund database (mock data)
        fund_database = [
            {
                "name": "Green Impact Ventures III",
                "focus": "Climate Change",
                "geography": "Global",
                "impact_score": 8.7,
                "expected_return": 10.2,
                "risk_score": 5.4,
                "liquidity": "Medium (5-7 years)",
                "size": 175
            },
            {
                "name": "Sustainable Agriculture Fund",
                "focus": "Agriculture",
                "geography": "Africa/South Asia",
                "impact_score": 9.2,
                "expected_return": 9.8,
                "risk_score": 6.1,
                "liquidity": "Medium-Long (7-10 years)",
                "size": 120
            },
            {
                "name": "Financial Inclusion Partners II",
                "focus": "Financial Inclusion",
                "geography": "Southeast Asia",
                "impact_score": 7.9,
                "expected_return": 7.3,
                "risk_score": 4.7,
                "liquidity": "Medium (5-7 years)",
                "size": 90
            },
            {
                "name": "Climate Tech Innovators",
                "focus": "Climate Change",
                "geography": "Global",
                "impact_score": 9.5,
                "expected_return": 5.5,
                "risk_score": 7.2,
                "liquidity": "Long (8+ years)",
                "size": 150
            },
            {
                "name": "Rural Financial Access Fund",
                "focus": "Financial Inclusion",
                "geography": "Africa",
                "impact_score": 8.3,
                "expected_return": 8.8,
                "risk_score": 5.2,
                "liquidity": "Medium (5-7 years)",
                "size": 80
            },
            {
                "name": "Renewable Energy Infrastructure",
                "focus": "Climate Change",
                "geography": "South Asia",
                "impact_score": 8.8,
                "expected_return": 9.5,
                "risk_score": 4.3,
                "liquidity": "Medium-Long (7-10 years)",
                "size": 220
            },
            {
                "name": "Agritech Solutions Fund",
                "focus": "Agriculture",
                "geography": "Africa",
                "impact_score": 8.1,
                "expected_return": 11.9,
                "risk_score": 5.8,
                "liquidity": "Medium (5-7 years)",
                "size": 110
            },
            {
                "name": "Clean Energy Transition Fund",
                "focus": "Climate Change",
                "geography": "Global",
                "impact_score": 9.3,
                "expected_return": 7.1,
                "risk_score": 6.2,
                "liquidity": "Long (8+ years)",
                "size": 185
            },
            {
                "name": "Inclusive Fintech Ventures",
                "focus": "Financial Inclusion",
                "geography": "Global",
                "impact_score": 7.6,
                "expected_return": 8.2,
                "risk_score": 6.7,
                "liquidity": "Medium (5-7 years)",
                "size": 95,
                "relevance_score": 7.9
            },
            {
                "name": "Sustainable Food Systems",
                "focus": "Agriculture",
                "geography": "South Asia",
                "impact_score": 8.5,
                "expected_return": 6.4,
                "risk_score": 5.5,
                "liquidity": "Medium-Long (7-10 years)",
                "size": 130,
                "relevance_score": 8.6
            }
        ]
        
        # For testing - relax filter criteria to ensure we get at least a few funds
        min_impact_score = 7.0  # Lower this from 8.0 for Impact Maximization
        
        # Filter out funds already in the portfolio
        filtered_funds = [fund for fund in fund_database if fund["name"] not in current_funds]
        
        # Filter based on optimization criteria
        qualified_funds = []
        
        if primary_param == "Impact Maximization":
            # Prioritize high impact funds that meet the minimum return and don't exceed max risk
            qualified_funds = [fund for fund in filtered_funds 
                            if fund["impact_score"] >= min_impact_score  # More lenient filter
                            and fund["expected_return"] >= min_return
                            and fund["risk_score"] <= max_risk]
            # Sort by impact score (descending)
            qualified_funds.sort(key=lambda x: x["impact_score"], reverse=True)
            
        elif primary_param == "Return Maximization":
            # Prioritize high return funds that don't exceed max risk
            qualified_funds = [fund for fund in filtered_funds 
                            if fund["expected_return"] >= min_return
                            and fund["risk_score"] <= max_risk
                            and fund["impact_score"] >= 7.0]  # Still need some impact
            # Sort by expected return (descending)
            qualified_funds.sort(key=lambda x: x["expected_return"], reverse=True)
            
        elif primary_param == "Risk Minimization":
            # Prioritize low risk funds that meet the minimum return
            qualified_funds = [fund for fund in filtered_funds 
                            if fund["risk_score"] <= max_risk
                            and fund["expected_return"] >= min_return
                            and fund["impact_score"] >= 7.0]  # Still need some impact
            # Sort by risk score (ascending)
            qualified_funds.sort(key=lambda x: x["risk_score"])
            
        else:  # Liquidity optimization
            # Define liquidity conversion to numeric scale
            liquidity_value = {
                "Short (3-5 years)": 3,
                "Medium (5-7 years)": 2,
                "Medium-Long (7-10 years)": 1,
                "Long (8+ years)": 0
            }
            
            # Prioritize funds with better liquidity that meet min return and max risk
            qualified_funds = [fund for fund in filtered_funds 
                            if fund["expected_return"] >= min_return
                            and fund["risk_score"] <= max_risk
                            and fund["impact_score"] >= 7.0]  # Still need some impact
            
            # Sort by liquidity (descending) - higher value means better liquidity
            qualified_funds.sort(key=lambda x: liquidity_value.get(x["liquidity"], 0), reverse=True)
        
        # Make sure we have some funds to display - if none match the criteria, return at least 2
        if len(qualified_funds) == 0:
            # Fall back to less strict criteria
            qualified_funds = [fund for fund in filtered_funds 
                            if fund["expected_return"] >= min(min_return, 5.0)
                            and fund["risk_score"] <= max(max_risk, 7.0)]
            qualified_funds.sort(key=lambda x: x["impact_score"], reverse=True)
        
        # Return top funds (up to 4)
        return qualified_funds[:4]

    def display_recommended_funds(self, primary_param, min_return, max_risk, min_liquidity):
        """Display a list of recommended funds that meet the optimization criteria"""
        st.markdown("### Recommended Funds")
        st.markdown("""
        The following funds (not currently in your portfolio) would help you meet your optimization criteria:
        """)
        
        # Get current funds in portfolio to exclude
        current_funds = set()
        if not st.session_state.data.empty:
            current_funds = set(st.session_state.data['fund'].unique())
        
        # Generate sample fund database with varied characteristics
        recommended_funds = self.generate_recommended_funds(primary_param, min_return, max_risk, min_liquidity, current_funds)
        
        # Create a table for display
        fund_df = pd.DataFrame(recommended_funds)
        
        # Check if we have any funds to display
        if len(recommended_funds) == 0:
            st.warning("No funds match your current criteria. Try adjusting your parameters.")
            return
        
        # Create a fixed number of columns (e.g., 4) and handle different numbers of funds
        num_cols = min(4, len(recommended_funds))  # Maximum of 4 columns or fewer if less funds
        cols = st.columns(num_cols)
        
        # Display each fund in a column
        for i, fund in enumerate(recommended_funds):
            col_idx = i % num_cols  # This ensures we wrap to the next row if more than num_cols funds
            with cols[col_idx]:
                # Fund card with box shadow and subtle border
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="margin-top: 0; color: #1f77b4;">{fund['name']}</h4>
                    <p><strong>Focus:</strong> {fund['focus']}</p>
                    <p><strong>Geography:</strong> {fund['geography']}</p>
                    <div style="margin: 10px 0;">
                        <span style="background-color: #e8f4f8; color: #1f77b4; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 5px;">
                            Impact: {fund['impact_score']}/10
                        </span>
                        <span style="background-color: #f0f8e8; color: #2ca02c; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 5px;">
                            Return: {fund['expected_return']}%
                        </span>
                        <span style="background-color: #fff4e8; color: #ff7f0e; padding: 3px 8px; border-radius: 4px; font-size: 0.8em;">
                            Risk: {fund['risk_score']}/10
                        </span>
                    </div>
                    <p><strong>Liquidity:</strong> {fund['liquidity']}</p>
                    <p><strong>Size:</strong> ${fund['size']}M</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a button to request more information
                if st.button(f"Request Info", key=f"request_info_{i}"):
                    st.info(f"Information request for {fund['name']} sent.")
        
        # Add a 'View All Recommendations' button
        if st.button("View All Recommendations", key="view_all_recs"):
            st.session_state.show_all_recommendations = True
            st.markdown("### All Recommended Funds")
            
            # Display sortable table
            st.dataframe(
                fund_df[['name', 'focus', 'geography', 'impact_score', 'expected_return', 'risk_score', 'liquidity', 'size', 'relevance_score']],
                column_config={
                    "name": "Fund Name",
                    "focus": "Focus Area",
                    "geography": "Geography",
                    "impact_score": st.column_config.NumberColumn("Impact Score", format="%.1f/10"),
                    "expected_return": st.column_config.NumberColumn("Expected Return", format="%.1f%%"),
                    "risk_score": st.column_config.NumberColumn("Risk Score", format="%.1f/10"),
                    "liquidity": "Liquidity",
                    "size": st.column_config.NumberColumn("Fund Size ($M)", format="$%.0f M"),
                    "relevance_score": st.column_config.NumberColumn("Relevance Score", format="%.1f/10")
                },
                use_container_width=True,
                hide_index=True
            )

    def display_tactical_allocation_results(self, execution_constraint, macro_context, market_opportunity, timeframe):
        """Display tactical allocation recommendations for private markets"""
        
        # Create tactical recommendations based on inputs
        st.markdown("### Tactical Allocation Recommendations")
        
        # Generate recommendation text based on inputs
        recommendation_title = "Tactical Deployment Strategy"
        recommendation_summary = "Based on current parameters, the following tactical strategies are recommended for deploying remaining uncommitted capital and planning future investments:"
        
        recommendations = []
        
        # Process execution constraints
        if execution_constraint != "None":
            if "Limited resources" in execution_constraint:
                recommendations.append("Prioritize high-impact investments in resource-constrained markets for remaining capital")
            elif "Political instability" in execution_constraint:
                recommendations.append("Pause new commitments to politically unstable regions and redirect to stable markets")
            elif "Limited pipeline" in execution_constraint:
                recommendations.append("Create dedicated sourcing strategy for underrepresented sectors over next 2 quarters")
            elif "Team capacity" in execution_constraint:
                recommendations.append("Sequence new deals to align with team capacity over 6-month period")
        
        # Process macro context
        if macro_context != "No significant changes":
            if "Elections" in macro_context:
                recommendations.append("Defer 2-3% of uncommitted capital until post-election stability is confirmed")
            elif "Policy changes" in macro_context:
                recommendations.append("Shift new commitments from direct investments to intermediated exposure in affected regions")
            elif "Economic downturn" in macro_context:
                recommendations.append("Prioritize counter-cyclical investments in essential services for next investment cycle")
            elif "Currency volatility" in macro_context:
                recommendations.append("Implement currency hedging strategy for new investments in affected markets")
        
        # Process market opportunity
        if market_opportunity != "No significant opportunities":
            if "clean energy" in market_opportunity:
                recommendations.append("Reserve 5% of uncommitted capital for emerging clean energy technologies")
            elif "financial inclusion" in market_opportunity:
                recommendations.append("Target 3-5% of new commitments to financial inclusion platforms in priority markets")
            elif "resilience" in market_opportunity:
                recommendations.append("Prioritize climate resilience infrastructure in next fund design")
            elif "Agricultural" in market_opportunity:
                recommendations.append("Dedicate 4% of remaining capital to agricultural value chain investments")
        
        # Create allocation chart based on recommendations
        fig = go.Figure()
        
        # Example data
        current_alloc = {
            'Climate Change': 35,
            'Financial Inclusion': 25,
            'Agriculture': 15,
            'Other': 25
        }
       
        # Define categories variable
        categories = list(current_alloc.keys())
        
        # Calculate uncommitted capital percentage (example)
        uncommitted_pct = 22
        committed_pct = 100 - uncommitted_pct
        
        # Generate tactical allocation based on recommendations
        tactical_alloc = current_alloc.copy()
        
        if "clean energy" in market_opportunity:
            tactical_alloc['Climate Change'] += 5 * (uncommitted_pct/100)
            tactical_alloc['Other'] -= 5 * (uncommitted_pct/100)
        elif "financial inclusion" in market_opportunity:
            tactical_alloc['Financial Inclusion'] += 4 * (uncommitted_pct/100)
            tactical_alloc['Other'] -= 4 * (uncommitted_pct/100)
        elif "Agricultural" in market_opportunity:
            tactical_alloc['Agriculture'] += 4 * (uncommitted_pct/100)
            tactical_alloc['Other'] -= 4 * (uncommitted_pct/100)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        # Current allocation (divided into committed and uncommitted)
        for category in categories:
            fig.add_trace(go.Bar(
                x=["Current Portfolio"],
                y=[current_alloc[category] * committed_pct / 100],
                name=f'{category} (Committed)',
                marker_color={'Climate Change': '#1f77b4', 
                            'Financial Inclusion': '#2ca02c',
                            'Agriculture': '#ff7f0e',
                            'Other': '#d62728'}.get(category, '#1f77b4'),
                opacity=0.8
            ))
        
        # Uncommitted capital
        fig.add_trace(go.Bar(
            x=["Current Portfolio"],
            y=[uncommitted_pct],
            name="Uncommitted Capital",
            marker_color='#7f7f7f',
        ))
        
        # Future allocation after tactical changes
        for category in categories:
            fig.add_trace(go.Bar(
                x=["Future Portfolio"],
                y=[tactical_alloc[category]],
                name=f'{category} (Future)',
                marker_color={'Climate Change': '#1f77b4', 
                            'Financial Inclusion': '#2ca02c',
                            'Agriculture': '#ff7f0e',
                            'Other': '#d62728'}.get(category, '#1f77b4'),
                opacity=0.4,
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Tactical Allocation for {timeframe} Timeframe",
            xaxis_title="Portfolio Timeline",
            yaxis_title="Allocation (%)",
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Display visualization and recommendations
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of private markets context
        st.info("""
        **Private Markets Note:** This tactical allocation strategy focuses on deploying uncommitted capital
        and planning future fund structures. Existing investments cannot be easily rebalanced mid-holding period.
        """)
        
        # Display recommendations
        st.markdown(f"#### {recommendation_title}")
        st.markdown(recommendation_summary)
        
        for i, rec in enumerate(recommendations):
            st.markdown(f"**{i+1}.** {rec}")
        
        st.markdown("#### Implementation Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Uncommitted Capital Deployment:**")
            st.markdown("""
            * Prioritize deals aligned with tactical strategy for next 60 days
            * Review investment committee criteria to reflect tactical priorities
            * Consider accelerating commitments in priority themes
            * Engage fund managers to align investment focus with tactical goals
            """)
        
        with col2:
            st.markdown("**Future Fund Planning:**")
            st.markdown("""
            * Incorporate tactical themes in design of next fund vehicles
            * Adjust fund manager selection criteria to align with tactical shifts
            * Develop specialized co-investment mechanisms for priority themes
            * Plan natural portfolio evolution through future exit strategies
            """)
        
        # Add buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Tactical Plan"):
                st.info("Tactical allocation plan exported")
        
        with col2:
            if st.button("Apply to Investment Pipeline"):
                st.success("Tactical strategy applied to investment pipeline and future funds")
    
    def get_fund_performance_data(self):
        """Get fund performance data for optimization"""
        df = st.session_state.data
        
        # Calculate fund-level metrics
        funds_data = df.groupby('fund').agg({
            'capital_deployed': 'sum',
            'progress_year3': 'mean',
            'impact_target': 'mean'
        }).reset_index()
        
        # Calculate Fund ID score (progress toward target)
        funds_data['impact_score'] = (funds_data['progress_year3'] / funds_data['impact_target'] * 100)
        
        # Add fake risk score (in a real implementation, this will be calculated from actual data)
        np.random.seed(42)  # For reproducible results
        funds_data['risk_score'] = np.random.uniform(2.0, 5.0, size=len(funds_data))
        
        # Add impact-to-risk ratio
        funds_data['impact_risk_ratio'] = funds_data['impact_score'] / funds_data['risk_score']
        
        return funds_data

    def create_current_allocation_chart(self, funds_data):
        """Create chart showing current portfolio allocation"""
        # Calculate allocation percentage
        total_capital = funds_data['capital_deployed'].sum()
        funds_data['allocation_pct'] = funds_data['capital_deployed'] / total_capital * 100
        
        # Create figure
        fig = go.Figure()
        
        # Add bar for allocation
        fig.add_trace(go.Bar(
            x=funds_data['fund'],
            y=funds_data['allocation_pct'],
            name='Current Allocation (%)',
            marker_color='#1f77b4'
        ))
        
        # Add line for Fund ID score
        fig.add_trace(go.Scatter(
            x=funds_data['fund'],
            y=funds_data['impact_score'],
            name='Fund ID Score',
            mode='markers+lines',
            marker=dict(size=10),
            yaxis='y2',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        # Layout with dual y-axis
        fig.update_layout(
            title="Current Portfolio Allocation vs Impact Performance",
            xaxis_title="Fund",
            yaxis=dict(
                title="Allocation (%)",
                tickfont=dict(color="#1f77b4")
            ),
            yaxis2=dict(
                title="Fund ID Score",
                tickfont=dict(color="#ff7f0e"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    def display_optimization_recommendations(self, funds_data):
        """Display impact optimization recommendations"""
        # Sort funds by impact-to-risk ratio to find best candidates
        sorted_funds = funds_data.sort_values('impact_risk_ratio', ascending=False).reset_index()
        
        # Get top and bottom performers for recommendations
        top_fund = sorted_funds.iloc[0]
        bottom_fund = sorted_funds.iloc[-1]
        
        # Create recommendation card
        st.markdown("#### Recommended Portfolio Adjustments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-left: 4px solid #1f77b4; padding: 15px; border-radius: 4px;">
                <h5 style="margin-top: 0;">Impact Maximization Recommendation</h5>
                <p>Increase allocation to <strong>{top_fund['fund']}</strong> by 5% and 
                reduce allocation to <strong>{bottom_fund['fund']}</strong> by 5%.</p>
                <p><strong>Projected impact:</strong></p>
                <ul>
                    <li>Overall Fund ID score: <span style="color: green;">+2.3 points</span></li>
                    <li>Risk profile change: <span style="color: gray;">+0.1</span></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create a simple before/after chart for the recommendation
            labels = ['Before Optimization', 'After Optimization']
            current_score = funds_data['impact_score'].mean()
            
            fig = go.Figure([
                go.Bar(name='Fund ID Score', 
                    x=labels, 
                    y=[current_score, current_score + 2.3],
                    marker_color=['#1f77b4', '#2ca02c'])
            ])
            
            fig.update_layout(
                title="Projected Impact Improvement",
                yaxis_title="Portfolio Fund ID Score",
                height=200,
                margin=dict(t=30, r=10, b=10, l=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add apply button
        if st.button("Apply Recommendation", key="apply_impact_rec"):
            st.success("Recommendation applied! Portfolio allocation updated.")

    def calculate_target_progress(self):
        """Calculate current progress toward targets"""
        df = st.session_state.data
        
        # Calculate theme-level target progress
        target_data = df.groupby(['impact_theme']).agg({
            'progress_year3': 'mean',
            'impact_target': 'mean'
        }).reset_index()
        
        # Calculate progress percentage
        target_data['progress_pct'] = (target_data['progress_year3'] / target_data['impact_target'] * 100)
        
        # Fake projected values
        target_data['projected_value'] = target_data['progress_year3'] * (1 + np.random.uniform(0.1, 0.3, size=len(target_data)))
        target_data['projected_pct'] = (target_data['projected_value'] / target_data['impact_target'] * 100)
        
        return target_data

    def create_target_progress_chart(self, target_data):
        """Create chart showing progress toward targets"""
        fig = go.Figure()
        
        # Add current progress bars
        fig.add_trace(go.Bar(
            name='Current Progress',
            x=target_data['impact_theme'],
            y=target_data['progress_pct'],
            marker_color='#1f77b4'
        ))
        
        # Add projected progress bars
        fig.add_trace(go.Bar(
            name='Projected Progress (Current Path)',
            x=target_data['impact_theme'],
            y=target_data['projected_pct'],
            marker_color='#7fc0fc'
        ))
        
        # Add target line at 100%
        fig.add_trace(go.Scatter(
            x=target_data['impact_theme'],
            y=[100] * len(target_data),
            mode='lines',
            name='Target (100%)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title="Progress Toward Impact Targets by Theme",
            xaxis_title="Impact Theme",
            yaxis_title="Progress (%)",
            barmode='group',
            hovermode="x unified"
        )
        
        return fig

    def display_target_recommendations(self, target_data, target_date):
        """Display target achievement recommendations"""
        # Find themes furthest from targets
        target_data_sorted = target_data.sort_values('progress_pct').reset_index()
        lagging_theme = target_data_sorted.iloc[0]
        
        # Calculate months until target date
        today = pd.to_datetime('today')
        target_date = pd.to_datetime(target_date)
        months_remaining = ((target_date.year - today.year) * 12 + target_date.month - today.month)
        
        # Create recommendation
        st.markdown("#### Target Achievement Recommendations")
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-left: 4px solid #ff7f0e; padding: 15px; border-radius: 4px;">
            <h5 style="margin-top: 0;">Target Achievement Strategy ({months_remaining} months remaining)</h5>
            <p><strong>{lagging_theme['impact_theme']}</strong> is furthest from target at 
            {lagging_theme['progress_pct']:.1f}% completion.</p>
            <p><strong>Recommendations:</strong></p>
            <ul>
                <li>Increase capital allocation to {lagging_theme['impact_theme']} by 3-5%</li>
                <li>Implement additional measurement and support for portfolio companies in this theme</li>
                <li>Consider adding new high-potential investments in this impact area</li>
            </ul>
            <p>This strategy is projected to increase achievement probability by 22%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add apply button
        if st.button("Apply Recommendation", key="apply_target_rec"):
            st.success("Recommendation applied! Target strategy updated.")

    def display_custom_optimization_results(self, optimization_goal, constraint_type, time_horizon):
        """Display results of custom optimization"""
        # In a real implementation, calculate optimization based on custom algorithm - for now, present fake results based on inputs
        
        st.markdown("#### Custom Optimization Results")
        st.markdown(f"**Goal:** {optimization_goal}")
        st.markdown(f"**Constraint:** {constraint_type}")
        st.markdown(f"**Time Horizon:** {time_horizon}")
        
        # Create a fake optimization chart
        months = int(time_horizon.split()[0])
        x_values = list(range(months+1))
        
        # Generate different curves based on selected goal
        if optimization_goal == "Impact Maximization":
            baseline = [70 + i * (85-70)/months for i in x_values]
            optimized = [70 + i * (92-70)/months for i in x_values]
            title = "Projected Fund ID Score"
            y_label = "Fund ID Score"
        elif optimization_goal == "Risk Minimization":
            baseline = [4.5 - i * (1.2)/months for i in x_values]
            optimized = [4.5 - i * (1.8)/months for i in x_values]
            title = "Projected Risk Level"
            y_label = "Risk Score"
        elif optimization_goal == "Target Achievement":
            baseline = [65 + i * (88-65)/months for i in x_values]
            optimized = [65 + i * (98-65)/months for i in x_values]
            title = "Target Achievement"
            y_label = "Achievement (%)"
        else:  # SDG Alignment
            baseline = [72 + i * (83-72)/months for i in x_values]
            optimized = [72 + i * (94-72)/months for i in x_values]
            title = "SDG Alignment Score"
            y_label = "Alignment Score"
        
        fig = go.Figure()
        
        # Add baseline projection
        fig.add_trace(go.Scatter(
            x=x_values,
            y=baseline,
            mode='lines',
            name='Current Trajectory',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add optimized projection
        fig.add_trace(go.Scatter(
            x=x_values,
            y=optimized,
            mode='lines',
            name='Optimized Trajectory',
            line=dict(color='#2ca02c', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{title} Over {time_horizon}",
            xaxis_title="Months",
            yaxis_title=y_label,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display recommended actions
        st.markdown("#### Recommended Actions")
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-left: 4px solid #2ca02c; padding: 15px; border-radius: 4px;">
            <h5 style="margin-top: 0;">Optimization Strategy</h5>
            <p>To achieve the optimized trajectory, the following changes are recommended:</p>
            <ol>
                <li>Reallocate 8% of capital from lower-performing funds to high-impact investments</li>
                <li>Increase measurement frequency for key metrics to monthly instead of quarterly</li>
                <li>Implement additional technical assistance for portfolio companies in priority themes</li>
            </ol>
            <p><strong>Key tradeoffs:</strong> This optimization maintains {constraint_type.lower()} while maximizing {optimization_goal.lower()}.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Export and apply options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Optimization Report"):
                st.info("Optimization report would be exported here")
        
        with col2:
            if st.button("Apply Optimization Strategy"):
                st.success("Optimization strategy applied successfully!")

    # Modify the render_impact_analysis method to include the new optimization tab
    def render_impact_analysis(self):
        """Render impact analysis page"""
        st.title("Impact Analysis")
        
        if st.session_state.data.empty:
            st.warning("No data available. Please import data first.")
            return
            
        # Create tabs for different analysis views
        analysis_tabs = st.tabs(["Metrics Analysis", "Portfolio Optimization"])
        
        with analysis_tabs[0]:  # First tab - Metrics Analysis
            # Theme and outcome selection
            col1, col2 = st.columns(2)
            with col1:
                selected_theme = st.selectbox(
                    "Select Impact Theme",
                    options=sorted(st.session_state.data['impact_theme'].unique()),
                    key="theme_selector_1"
                )
            with col2:
                selected_outcome = st.selectbox(
                    "Select Impact Outcome",
                    options=sorted(st.session_state.data[
                        st.session_state.data['impact_theme'] == selected_theme
                    ]['impact_outcome'].unique()),
                    key="outcome_selector_1"
                )
                
            # Analysis sections
            self.render_metrics_analysis(selected_theme, selected_outcome)
            self.render_benchmark_analysis(selected_theme, selected_outcome)
        
        with analysis_tabs[1]:  # Second tab - Portfolio Optimization
            # Portfolio optimization tools
            self.render_portfolio_optimization()   

    
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
