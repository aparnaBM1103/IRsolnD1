import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import io

class ImpactDataUtils:
    @staticmethod
    def process_uploaded_file(file) -> pd.DataFrame:
        """Process uploaded data file"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
                
            required_cols = [
                'fund', 'company', 'impact_theme', 'impact_outcome',
                'impact_kpi', 'impact_baseline', 'impact_target'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
                
            return df
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict:
        """Validate imported data"""
        validation = {
            'completeness': {},
            'quality': {},
            'issues': []
        }
        
        # Required columns
        required_cols = [
            'fund', 'company', 'impact_theme', 'impact_outcome',
            'impact_kpi', 'impact_baseline', 'impact_target'
        ]
        
        # Check completeness
        for col in required_cols:
            if col not in df.columns:
                validation['issues'].append(f"Missing column: {col}")
            else:
                completeness = (df[col].notna().sum() / len(df)) * 100
                validation['completeness'][col] = completeness
                
                if completeness < 75:
                    validation['issues'].append(
                        f"Low completeness for {col}: {completeness:.1f}%"
                    )
        
        # Check data quality
        if all(col in df.columns for col in ['impact_baseline', 'impact_target']):
            valid_targets = (df['impact_target'] > df['impact_baseline']).sum() / len(df) * 100
            validation['quality']['valid_targets'] = valid_targets
            
            if valid_targets < 90:
                validation['issues'].append(
                    f"Some targets not higher than baselines: {valid_targets:.1f}%"
                )
        
        # Check progress tracking
        progress_cols = [col for col in df.columns if 'progress_year' in col]
        if progress_cols:
            progress_tracking = df[progress_cols].notna().all(axis=1).sum() / len(df) * 100
            validation['quality']['progress_tracking'] = progress_tracking
            
            if progress_tracking < 75:
                validation['issues'].append(
                    f"Low progress tracking: {progress_tracking:.1f}%"
                )
        
        return validation

    @staticmethod
    def calculate_portfolio_metrics(df: pd.DataFrame) -> Dict:
        """Calculate portfolio-level metrics"""
        metrics = {}
        
        # Fund-level metrics
        fund_metrics = df.groupby('fund').agg({
            'company': lambda x: len(set(x)) >= 3,
            'impact_baseline': lambda x: x.notna().mean() >= 0.75,
            'impact_target': lambda x: x.notna().mean() >= 0.75,
            'relevance': lambda x: (x == 'High').mean() >= 0.75
        })
        
        total_funds = len(fund_metrics)
        
        metrics['tracking_coverage'] = (fund_metrics['company'].sum() / total_funds) * 100
        metrics['baseline_coverage'] = (fund_metrics['impact_baseline'].sum() / total_funds) * 100
        metrics['portfolio_targets'] = (fund_metrics['impact_target'].sum() / total_funds) * 100
        
        # Company targets
        company_targets = df.groupby(['fund', 'company'])['impact_target'].apply(
            lambda x: x.notna().any()
        ).groupby('fund').mean()
        metrics['company_targets'] = (company_targets >= 0.75).mean() * 100
        
        # High relevance metrics
        metrics['high_relevance'] = (fund_metrics['relevance'].sum() / total_funds) * 100
        
        # YoY progress
        progress_cols = ['progress_year1', 'progress_year2', 'progress_year3']
        yoy_tracking = df.groupby('fund')[progress_cols].apply(
            lambda x: x.notna().all(axis=1).mean() >= 0.75
        )
        metrics['yoy_progress'] = (yoy_tracking.sum() / total_funds) * 100
        
        return metrics

    @staticmethod
    def get_template_file() -> bytes:
        """Generate template file"""
        df = pd.DataFrame({
            'fund': ['Example Fund'],
            'company': ['Example Company'],
            'impact_theme': ['Climate Action'],
            'impact_outcome': ['GHG Reduction'],
            'impact_kpi': ['CO2 Emissions Avoided'],
            'impact_baseline': [1000],
            'impact_target': [5000],
            'progress_year1': [2000],
            'progress_year2': [3000],
            'progress_year3': [4000],
            'relevance': ['High'],
            'geography': ['Global'],
            'capital_invested': [1000000],
            'sdg': ['SDG13']
        })
        
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        return buffer.getvalue()

    @staticmethod
    def aggregate_impact_data(df: pd.DataFrame, 
                            level: str = 'portfolio',
                            filters: Optional[Dict] = None) -> pd.DataFrame:
        """Aggregate impact data at specified level"""
        if filters:
            df = df.copy()
            for key, value in filters.items():
                if value:
                    df = df[df[key].isin(value)]
        
        if level == 'portfolio':
            return df.groupby(['impact_theme', 'impact_outcome']).agg({
                'fund': lambda x: ', '.join(sorted(set(x))),
                'company': 'nunique',
                'capital_invested': 'sum',
                'progress_year1': 'mean',
                'progress_year2': 'mean',
                'progress_year3': 'mean',
                'impact_target': 'mean'
            }).reset_index()
            
        elif level == 'fund':
            return df.groupby(['fund', 'impact_kpi']).agg({
                'company': lambda x: ', '.join(sorted(set(x))),
                'capital_invested': 'sum',
                'progress_year1': 'mean',
                'progress_year2': 'mean',
                'progress_year3': 'mean',
                'impact_target': 'mean'
            }).reset_index()
            
        else:  # company level
            return df