"""
HTML report generator for forensic analysis results.
"""
import json
from typing import Dict, Any, List
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


class ForensicReportGenerator:
    """Generate comprehensive HTML reports for forensic analysis."""
    
    def __init__(self):
        self.report_template = self._load_template()
        self.chart_config = {
            'height': 400,
            'template': 'plotly_white',
            'font_size': 12
        }
    
    def generate_report(self, analysis_results: Dict[str, Any], 
                       forensic_profile: Dict[str, Any],
                       output_path: str = None) -> str:
        """
        Generate comprehensive HTML forensic report.
        
        Args:
            analysis_results: Complete analysis results
            forensic_profile: Generated forensic profile
            output_path: Output file path (optional)
            
        Returns:
            Path to generated report file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"forensic_report_{timestamp}.html"
        
        # Generate report sections
        report_data = {
            'metadata': self._generate_metadata_section(analysis_results, forensic_profile),
            'executive_summary': self._generate_executive_summary(forensic_profile),
            'user_profile': self._generate_user_profile_section(forensic_profile),
            'behavioral_analysis': self._generate_behavioral_section(analysis_results),
            'temporal_analysis': self._generate_temporal_section(analysis_results),
            'risk_assessment': self._generate_risk_section(forensic_profile),
            'technical_details': self._generate_technical_section(analysis_results),
            'charts': self._generate_charts(analysis_results),
            'recommendations': self._generate_recommendations_section(forensic_profile)
        }
        
        # Generate HTML report
        html_content = self._build_html_report(report_data)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Forensic report generated: {output_path}")
        return output_path
    
    def _load_template(self) -> str:
        """Load HTML template for the report."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Girlfriend App - Forensic Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }
                .header {
                    text-align: center;
                    border-bottom: 3px solid #2c3e50;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                }
                .header h1 {
                    color: #2c3e50;
                    margin-bottom: 10px;
                }
                .section {
                    margin-bottom: 40px;
                }
                .section h2 {
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-bottom: 20px;
                }
                .section h3 {
                    color: #2c3e50;
                    margin-top: 25px;
                    margin-bottom: 15px;
                }
                .grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .card {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }
                .risk-high { border-left-color: #e74c3c; }
                .risk-moderate { border-left-color: #f39c12; }
                .risk-low { border-left-color: #27ae60; }
                .risk-critical { border-left-color: #8e44ad; }
                .stat-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }
                .stat-item {
                    background-color: white;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #dee2e6;
                    text-align: center;
                }
                .stat-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }
                .stat-label {
                    font-size: 14px;
                    color: #6c757d;
                    margin-top: 5px;
                }
                .chart-container {
                    margin: 20px 0;
                    padding: 15px;
                    background-color: white;
                    border-radius: 8px;
                    border: 1px solid #dee2e6;
                }
                .metadata-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                }
                .metadata-item {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                }
                .alert {
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 8px;
                    border-left: 4px solid;
                }
                .alert-danger {
                    background-color: #f8d7da;
                    border-left-color: #dc3545;
                    color: #721c24;
                }
                .alert-warning {
                    background-color: #fff3cd;
                    border-left-color: #ffc107;
                    color: #856404;
                }
                .alert-info {
                    background-color: #d1ecf1;
                    border-left-color: #17a2b8;
                    color: #0c5460;
                }
                .timeline {
                    position: relative;
                    margin: 20px 0;
                }
                .timeline-item {
                    margin-bottom: 20px;
                    padding-left: 30px;
                    position: relative;
                }
                .timeline-item::before {
                    content: '';
                    position: absolute;
                    left: 0;
                    top: 5px;
                    width: 12px;
                    height: 12px;
                    background-color: #3498db;
                    border-radius: 50%;
                }
                .footer {
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #dee2e6;
                    text-align: center;
                    color: #6c757d;
                    font-size: 14px;
                }
                @media print {
                    body { background-color: white; }
                    .container { box-shadow: none; }
                    .grid { grid-template-columns: 1fr; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                {content}
            </div>
        </body>
        </html>
        """
    
    def _generate_metadata_section(self, analysis_results: Dict[str, Any], 
                                  forensic_profile: Dict[str, Any]) -> str:
        """Generate metadata section of the report."""
        metadata_analysis = analysis_results.get('metadata_analysis', {})
        profile_metadata = forensic_profile.get('profile_metadata', {})
        
        db_info = metadata_analysis.get('database_info', {})
        app_detection = metadata_analysis.get('app_detection', {})
        
        return f"""
        <div class="section">
            <h2>Case Information</h2>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>Database Type:</strong><br>
                    {db_info.get('type', 'Unknown')}
                </div>
                <div class="metadata-item">
                    <strong>Identified Application:</strong><br>
                    {app_detection.get('detected_app', 'Unknown')} 
                    ({app_detection.get('confidence', 0):.1%} confidence)
                </div>
                <div class="metadata-item">
                    <strong>Total Records:</strong><br>
                    {db_info.get('total_records', 0):,}
                </div>
                <div class="metadata-item">
                    <strong>Analysis Date:</strong><br>
                    {profile_metadata.get('profile_generated', 'Unknown')[:19]}
                </div>
                <div class="metadata-item">
                    <strong>Data Quality Score:</strong><br>
                    {profile_metadata.get('data_quality_score', 0):.2f} / 1.0
                </div>
                <div class="metadata-item">
                    <strong>Analysis Completeness:</strong><br>
                    {profile_metadata.get('analysis_completeness', 0):.1%}
                </div>
            </div>
        </div>
        """
    
    def _generate_executive_summary(self, forensic_profile: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        risk_factors = forensic_profile.get('risk_factors', {})
        user_characteristics = forensic_profile.get('user_characteristics', {})
        investigation_insights = forensic_profile.get('investigation_insights', {})
        
        overall_risk = risk_factors.get('overall_risk_level', 'unknown')
        risk_class = f"risk-{overall_risk.lower()}"
        
        key_findings = investigation_insights.get('key_findings', [])
        behavioral_summary = investigation_insights.get('behavioral_summary', 'No summary available')
        
        alert_class = {
            'high': 'alert-danger',
            'critical': 'alert-danger',
            'moderate': 'alert-warning',
            'low': 'alert-info'
        }.get(overall_risk.lower(), 'alert-info')
        
        findings_html = ""
        if key_findings:
            findings_html = "<ul>" + "".join(f"<li>{finding}</li>" for finding in key_findings[:5]) + "</ul>"
        
        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            
            <div class="alert {alert_class}">
                <strong>Overall Risk Assessment: {overall_risk.upper()}</strong><br>
                {behavioral_summary}
            </div>
            
            <div class="grid">
                <div class="card {risk_class}">
                    <h3>Risk Profile</h3>
                    <div class="stat-value">{overall_risk.upper()}</div>
                    <div class="stat-label">Overall Risk Level</div>
                </div>
                <div class="card">
                    <h3>Communication Style</h3>
                    <div class="stat-value">{user_characteristics.get('communication_style', 'Unknown')}</div>
                    <div class="stat-label">Interaction Pattern</div>
                </div>
            </div>
            
            {f'<h3>Key Findings</h3>{findings_html}' if findings_html else ''}
        </div>
        """
    
    def _generate_user_profile_section(self, forensic_profile: Dict[str, Any]) -> str:
        """Generate user profile section."""
        user_characteristics = forensic_profile.get('user_characteristics', {})
        psychological_indicators = forensic_profile.get('psychological_indicators', {})
        behavioral_patterns = forensic_profile.get('behavioral_patterns', {})
        
        personality_traits = user_characteristics.get('personality_traits', [])
        interests = user_characteristics.get('interests_and_topics', [])
        emotional_state = psychological_indicators.get('emotional_state', 'unknown')
        social_needs = psychological_indicators.get('social_needs', [])
        
        return f"""
        <div class="section">
            <h2>User Profile Analysis</h2>
            
            <div class="grid">
                <div class="card">
                    <h3>Psychological Profile</h3>
                    <p><strong>Emotional State:</strong> {emotional_state.title()}</p>
                    <p><strong>Social Needs:</strong> {', '.join(social_needs) if social_needs else 'None identified'}</p>
                    <p><strong>Language Complexity:</strong> {user_characteristics.get('language_complexity', 0):.2f}</p>
                </div>
                <div class="card">
                    <h3>Behavioral Characteristics</h3>
                    <p><strong>Usage Patterns:</strong> {behavioral_patterns.get('usage_patterns', {})}</p>
                    <p><strong>Engagement Level:</strong> {behavioral_patterns.get('engagement_level', 'unknown').title()}</p>
                    <p><strong>Activity Rhythm:</strong> {behavioral_patterns.get('activity_rhythm', {})}</p>
                </div>
            </div>
            
            {self._generate_interests_section(interests)}
        </div>
        """
    
    def _generate_interests_section(self, interests: List[str]) -> str:
        """Generate interests and topics section."""
        if not interests:
            return ""
        
        interests_html = " • ".join(interests[:10])  # Top 10 interests
        
        return f"""
        <div class="card">
            <h3>Identified Interests & Topics</h3>
            <p>{interests_html}</p>
        </div>
        """
    
    def _generate_behavioral_section(self, analysis_results: Dict[str, Any]) -> str:
        """Generate behavioral analysis section."""
        behavioral_analysis = analysis_results.get('behavioral_analysis', {})
        content_analysis = analysis_results.get('content_analysis', {})
        
        # Aggregate behavioral data
        total_interactions = 0
        avg_message_length = 0
        interaction_count = 0
        
        for table_results in behavioral_analysis.values():
            interaction_freq = table_results.get('interaction_frequency', {})
            total_interactions += interaction_freq.get('avg_daily_interactions', 0)
            
            comm_style = table_results.get('communication_style', {})
            if 'avg_message_length' in comm_style:
                avg_message_length += comm_style['avg_message_length']
                interaction_count += 1
        
        if interaction_count > 0:
            avg_message_length /= interaction_count
        
        # Aggregate sentiment data
        positive_sentiment = 0
        negative_sentiment = 0
        sentiment_count = 0
        
        for table_results in content_analysis.values():
            for column_results in table_results.values():
                sentiment = column_results.get('sentiment_analysis', {})
                overall_sentiment = sentiment.get('overall_sentiment', {})
                if overall_sentiment:
                    positive_sentiment += overall_sentiment.get('positive', 0)
                    negative_sentiment += overall_sentiment.get('negative', 0)
                    sentiment_count += 1
        
        if sentiment_count > 0:
            positive_sentiment /= sentiment_count
            negative_sentiment /= sentiment_count
        
        return f"""
        <div class="section">
            <h2>Behavioral Analysis</h2>
            
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="stat-value">{total_interactions:.1f}</div>
                    <div class="stat-label">Avg Daily Interactions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{avg_message_length:.0f}</div>
                    <div class="stat-label">Avg Message Length</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{positive_sentiment:.1%}</div>
                    <div class="stat-label">Positive Sentiment</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{negative_sentiment:.1%}</div>
                    <div class="stat-label">Negative Sentiment</div>
                </div>
            </div>
            
            {self._generate_communication_analysis(behavioral_analysis)}
        </div>
        """
    
    def _generate_communication_analysis(self, behavioral_analysis: Dict[str, Any]) -> str:
        """Generate communication analysis subsection."""
        comm_patterns = []
        
        for table_name, table_results in behavioral_analysis.items():
            comm_style = table_results.get('communication_style', {})
            if comm_style:
                question_freq = comm_style.get('question_frequency', 0)
                exclamation_freq = comm_style.get('exclamation_frequency', 0)
                
                comm_patterns.append(f"""
                <div class="card">
                    <h3>Communication Style ({table_name})</h3>
                    <p><strong>Question Frequency:</strong> {question_freq:.2%}</p>
                    <p><strong>Exclamation Usage:</strong> {exclamation_freq:.2%}</p>
                    <p><strong>Capital Letter Usage:</strong> {comm_style.get('capital_usage', 0):.2%}</p>
                </div>
                """)
        
        return "".join(comm_patterns)
    
    def _generate_temporal_section(self, analysis_results: Dict[str, Any]) -> str:
        """Generate temporal analysis section."""
        temporal_analysis = analysis_results.get('temporal_analysis', {})
        
        if not temporal_analysis:
            return ""
        
        # Get first temporal analysis for display
        first_analysis = next(iter(temporal_analysis.values()), {})
        
        activity_patterns = first_analysis.get('activity_patterns', {})
        circadian_patterns = first_analysis.get('circadian_patterns', {})
        usage_intensity = first_analysis.get('usage_intensity', {})
        
        peak_hour = activity_patterns.get('peak_activity_hour', 12)
        chronotype = circadian_patterns.get('chronotype', 'unknown')
        daily_stats = usage_intensity.get('daily_stats', {})
        
        return f"""
        <div class="section">
            <h2>Temporal Behavior Analysis</h2>
            
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="stat-value">{peak_hour}:00</div>
                    <div class="stat-label">Peak Activity Hour</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{chronotype.replace('_', ' ').title()}</div>
                    <div class="stat-label">Chronotype</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{daily_stats.get('average_daily_messages', 0):.1f}</div>
                    <div class="stat-label">Daily Message Average</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{daily_stats.get('active_days', 0)}</div>
                    <div class="stat-label">Active Days</div>
                </div>
            </div>
            
            {self._generate_activity_breakdown(activity_patterns)}
        </div>
        """
    
    def _generate_activity_breakdown(self, activity_patterns: Dict[str, Any]) -> str:
        """Generate activity breakdown subsection."""
        time_breakdown = activity_patterns.get('time_period_breakdown', {})
        
        if not time_breakdown:
            return ""
        
        periods = []
        for period, data in time_breakdown.items():
            percentage = data.get('percentage', 0)
            periods.append(f"""
            <div class="stat-item">
                <div class="stat-value">{percentage:.1f}%</div>
                <div class="stat-label">{period.title()} Activity</div>
            </div>
            """)
        
        return f"""
        <h3>Activity Distribution by Time Period</h3>
        <div class="stat-grid">
            {''.join(periods)}
        </div>
        """
    
    def _generate_risk_section(self, forensic_profile: Dict[str, Any]) -> str:
        """Generate risk assessment section."""
        risk_factors = forensic_profile.get('risk_factors', {})
        
        overall_risk = risk_factors.get('overall_risk_level', 'unknown')
        identified_risks = risk_factors.get('identified_risks', [])
        concerning_patterns = risk_factors.get('concerning_patterns', [])
        recommendations = risk_factors.get('mitigation_recommendations', [])
        
        risk_class = f"risk-{overall_risk.lower()}"
        
        return f"""
        <div class="section">
            <h2>Risk Assessment</h2>
            
            <div class="card {risk_class}">
                <h3>Overall Risk Level: {overall_risk.upper()}</h3>
                <p>Based on comprehensive behavioral analysis across multiple dimensions.</p>
            </div>
            
            {self._generate_risk_factors_list(identified_risks, concerning_patterns)}
            {self._generate_recommendations_list(recommendations)}
        </div>
        """
    
    def _generate_risk_factors_list(self, identified_risks: List[str], concerning_patterns: List[str]) -> str:
        """Generate risk factors and concerning patterns lists."""
        risks_html = ""
        patterns_html = ""
        
        if identified_risks:
            risks_html = f"""
            <div class="card">
                <h3>Identified Risk Factors</h3>
                <ul>
                    {''.join(f'<li>{risk.replace("_", " ").title()}</li>' for risk in identified_risks)}
                </ul>
            </div>
            """
        
        if concerning_patterns:
            patterns_html = f"""
            <div class="card">
                <h3>Concerning Patterns</h3>
                <ul>
                    {''.join(f'<li>{pattern}</li>' for pattern in concerning_patterns)}
                </ul>
            </div>
            """
        
        return f"""
        <div class="grid">
            {risks_html}
            {patterns_html}
        </div>
        """
    
    def _generate_recommendations_list(self, recommendations: List[str]) -> str:
        """Generate recommendations list."""
        if not recommendations:
            return ""
        
        return f"""
        <div class="card">
            <h3>Recommendations</h3>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in recommendations)}
            </ul>
        </div>
        """
    
    def _generate_technical_section(self, analysis_results: Dict[str, Any]) -> str:
        """Generate technical details section."""
        metadata_analysis = analysis_results.get('metadata_analysis', {})
        
        db_info = metadata_analysis.get('database_info', {})
        technical_indicators = metadata_analysis.get('technical_indicators', {})
        
        return f"""
        <div class="section">
            <h2>Technical Analysis Details</h2>
            
            <div class="grid">
                <div class="card">
                    <h3>Database Information</h3>
                    <p><strong>File Type:</strong> {db_info.get('type', 'Unknown')}</p>
                    <p><strong>Total Tables:</strong> {db_info.get('total_tables', 0)}</p>
                    <p><strong>Total Records:</strong> {db_info.get('total_records', 0):,}</p>
                    <p><strong>File Size:</strong> {technical_indicators.get('database_size', 0):,} bytes</p>
                </div>
                <div class="card">
                    <h3>Analysis Metadata</h3>
                    <p><strong>Extraction Time:</strong> {db_info.get('extraction_time', 'Unknown')[:19]}</p>
                    <p><strong>App Confidence:</strong> {technical_indicators.get('app_confidence', 0):.1%}</p>
                    <p><strong>Record Count:</strong> {technical_indicators.get('record_count', 0):,}</p>
                </div>
            </div>
        </div>
        """
    
    def _generate_charts(self, analysis_results: Dict[str, Any]) -> str:
        """Generate charts section with embedded visualizations."""
        charts_html = []
        
        # Generate temporal activity chart
        temporal_chart = self._create_temporal_activity_chart(analysis_results)
        if temporal_chart:
            charts_html.append(f"""
            <div class="chart-container">
                <h3>Temporal Activity Patterns</h3>
                {temporal_chart}
            </div>
            """)
        
        # Generate sentiment chart
        sentiment_chart = self._create_sentiment_chart(analysis_results)
        if sentiment_chart:
            charts_html.append(f"""
            <div class="chart-container">
                <h3>Sentiment Analysis</h3>
                {sentiment_chart}
            </div>
            """)
        
        if not charts_html:
            return ""
        
        return f"""
        <div class="section">
            <h2>Data Visualizations</h2>
            {''.join(charts_html)}
        </div>
        """
    
    def _create_temporal_activity_chart(self, analysis_results: Dict[str, Any]) -> str:
        """Create temporal activity chart."""
        temporal_analysis = analysis_results.get('temporal_analysis', {})
        
        if not temporal_analysis:
            return ""
        
        # Get hourly distribution from first temporal analysis
        first_analysis = next(iter(temporal_analysis.values()), {})
        activity_patterns = first_analysis.get('activity_patterns', {})
        hourly_dist = activity_patterns.get('hourly_distribution', {})
        
        if not hourly_dist:
            return ""
        
        try:
            # Create plotly chart
            hours = list(range(24))
            activity = [hourly_dist.get(str(hour), 0) for hour in hours]
            
            fig = go.Figure(data=go.Scatter(
                x=hours,
                y=activity,
                mode='lines+markers',
                name='Activity Level',
                line=dict(color='#3498db', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="24-Hour Activity Pattern",
                xaxis_title="Hour of Day",
                yaxis_title="Message Count",
                template="plotly_white",
                height=300
            )
            
            return fig.to_html(include_plotlyjs=False, div_id="temporal_chart")
        
        except Exception as e:
            return f"<p>Error generating temporal chart: {e}</p>"
    
    def _create_sentiment_chart(self, analysis_results: Dict[str, Any]) -> str:
        """Create sentiment analysis chart."""
        content_analysis = analysis_results.get('content_analysis', {})
        
        if not content_analysis:
            return ""
        
        try:
            # Aggregate sentiment data
            sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
            count = 0
            
            for table_results in content_analysis.values():
                for column_results in table_results.values():
                    sentiment = column_results.get('sentiment_analysis', {})
                    overall_sentiment = sentiment.get('overall_sentiment', {})
                    if overall_sentiment:
                        sentiments['Positive'] += overall_sentiment.get('positive', 0)
                        sentiments['Negative'] += overall_sentiment.get('negative', 0)
                        sentiments['Neutral'] += overall_sentiment.get('neutral', 0)
                        count += 1
            
            if count == 0:
                return ""
            
            # Average the sentiments
            for key in sentiments:
                sentiments[key] /= count
            
            # Create pie chart
            fig = go.Figure(data=go.Pie(
                labels=list(sentiments.keys()),
                values=list(sentiments.values()),
                hole=0.3,
                marker_colors=['#27ae60', '#e74c3c', '#95a5a6']
            ))
            
            fig.update_layout(
                title="Overall Sentiment Distribution",
                template="plotly_white",
                height=300
            )
            
            return fig.to_html(include_plotlyjs=False, div_id="sentiment_chart")
        
        except Exception as e:
            return f"<p>Error generating sentiment chart: {e}</p>"
    
    def _generate_recommendations_section(self, forensic_profile: Dict[str, Any]) -> str:
        """Generate recommendations section."""
        investigation_insights = forensic_profile.get('investigation_insights', {})
        
        further_investigation = investigation_insights.get('further_investigation', [])
        evidence_strength = investigation_insights.get('evidence_strength', {})
        
        return f"""
        <div class="section">
            <h2>Investigation Recommendations</h2>
            
            <div class="grid">
                <div class="card">
                    <h3>Evidence Assessment</h3>
                    <p><strong>Overall Strength:</strong> {evidence_strength.get('overall_strength', 'unknown').title()}</p>
                    <p><strong>Data Quality:</strong> {evidence_strength.get('data_quality', 'unknown')}</p>
                    <p><strong>Analysis Completeness:</strong> {evidence_strength.get('analysis_completeness', 'unknown')}</p>
                </div>
                <div class="card">
                    <h3>Next Steps</h3>
                    {self._generate_next_steps_list(further_investigation)}
                </div>
            </div>
        </div>
        """
    
    def _generate_next_steps_list(self, further_investigation: List[str]) -> str:
        """Generate next steps list."""
        if not further_investigation:
            return "<p>No specific recommendations at this time.</p>"
        
        return f"""
        <ul>
            {''.join(f'<li>{step}</li>' for step in further_investigation)}
        </ul>
        """
    
    def _build_html_report(self, report_data: Dict[str, Any]) -> str:
        """Build the complete HTML report."""
        content = f"""
        <div class="header">
            <h1>AI Girlfriend Application - Forensic Analysis Report</h1>
            <p>Digital Forensics & Behavioral Analysis</p>
        </div>
        
        {report_data['metadata']}
        {report_data['executive_summary']}
        {report_data['user_profile']}
        {report_data['behavioral_analysis']}
        {report_data['temporal_analysis']}
        {report_data['risk_assessment']}
        {report_data['charts']}
        {report_data['technical_details']}
        {report_data['recommendations']}
        
        <div class="footer">
            <p>Report generated by AI Girlfriend Forensics Analyzer v1.0</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>CONFIDENTIAL:</strong> This report contains sensitive information and should be handled according to applicable privacy and legal requirements.</p>
        </div>
        """
        
        return self.report_template.replace("{content}", content)
