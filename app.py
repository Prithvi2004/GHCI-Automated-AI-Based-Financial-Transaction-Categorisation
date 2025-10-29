"""
Advanced Streamlit Web UI for Transaction Categorization
Modern, interactive interface with enhanced visualizations and UX
"""

import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src.preprocess import clean_text
from src.feedback import FeedbackSystem
from src.explainability import TransactionExplainer
import yaml
import json
import time


# Page configuration
st.set_page_config(
    page_title="AI Transaction Categorizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI-Powered Transaction Categorization System | F1-Score: 0.9831"
    }
)

# Advanced Custom CSS with animations and modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 0.8s ease-in;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: white;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .error-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stat-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        margin: 0.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .category-pill {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        background: #667eea;
        color: white;
        font-size: 0.9rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    
    .stButton>button {
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .confidence-high {
        color: #56ab2f;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .confidence-medium {
        color: #f2994a;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .confidence-low {
        color: #eb3349;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .example-chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        background: #f0f2f6;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .example-chip:hover {
        background: #667eea;
        color: white;
        border: 2px solid #764ba2;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model"""
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None


@st.cache_data
def load_config():
    """Load category configuration"""
    with open("config/categories.yaml", 'r') as f:
        return yaml.safe_load(f)


def main():
    # Animated header
    st.markdown('<div class="main-header">💰 AI Transaction Categorizer</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by Advanced Machine Learning | F1-Score: 0.9831</div>', 
                unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.markdown('<div class="error-box">❌ <b>Model not found.</b> Please train the model first by running <code>python train_model.py</code></div>', 
                   unsafe_allow_html=True)
        return
    
    config = load_config()
    categories = model.get_categories()
    feedback_system = FeedbackSystem()
    
    # Enhanced Sidebar with stats
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        
        page = st.radio("Navigation Menu", [
            "🔮 Smart Categorizer",
            "📊 Performance Analytics",
            "🎨 Category Designer",
            "💬 Feedback Hub",
            "� Insights Dashboard"
        ], label_visibility="collapsed")
        
        st.markdown("---")
        
        # Quick stats in sidebar
        st.markdown("### 📌 Quick Stats")
        st.markdown(f'<div class="stat-badge">✅ {len(categories)} Categories</div>', unsafe_allow_html=True)
        
        if os.path.exists("evaluation_report.json"):
            with open("evaluation_report.json", 'r') as f:
                report = json.load(f)
            st.markdown(f'<div class="stat-badge">🎯 F1: {report["macro_f1"]:.4f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-badge">📈 Acc: {report["classification_report"]["accuracy"]:.4f}</div>', unsafe_allow_html=True)
        
        stats = feedback_system.get_feedback_stats()
        if stats['total_feedback'] > 0:
            st.markdown(f'<div class="stat-badge">💬 {stats["total_feedback"]} Feedbacks</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model info
        with st.expander("ℹ️ Model Info"):
            st.write("**Architecture:** Ensemble")
            st.write("**Algorithms:** RF + GB + LR")
            st.write("**Features:** TF-IDF")
            st.write("**Training:** 928 samples")
            st.write("**Testing:** 232 samples")
        
        # Quick tips
        with st.expander("💡 Pro Tips"):
            st.write("• Upload CSV for batch processing")
            st.write("• Provide feedback for low confidence")
            st.write("• Check Analytics for insights")
            st.write("• Download results as CSV")
    
    # Main content based on page selection
    if page == "🔮 Smart Categorizer":
        show_categorization_page(model, categories, feedback_system)
    
    elif page == "📊 Performance Analytics":
        show_performance_page(model)
    
    elif page == "🎨 Category Designer":
        show_category_management_page(config)
    
    elif page == "💬 Feedback Hub":
        show_feedback_page(feedback_system)
    
    elif page == "📈 Insights Dashboard":
        show_analytics_dashboard(model)


def show_categorization_page(model, categories, feedback_system):
    """Enhanced transaction categorization interface"""
    st.markdown("## 🔮 Smart Transaction Categorizer")
    st.markdown("Instantly categorize transactions using AI-powered analysis with 98.31% accuracy")
    
    # Tab navigation for cleaner interface
    tab1, tab2 = st.tabs(["🎯 Single Transaction", "📁 Batch Processing"])
    
    with tab1:
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            st.markdown("### Enter Transaction Details")
            
            # Initialize session state for example selection
            if 'selected_example' not in st.session_state:
                st.session_state.selected_example = ""
            
            # Use the selected example as default value
            transaction_text = st.text_input(
                "Transaction Description",
                value=st.session_state.selected_example,
                placeholder="e.g., STARBUCKS STORE #12345 NEW YORK",
                help="Enter the merchant name or transaction description",
                label_visibility="collapsed"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                amount = st.number_input("💵 Amount (Optional)", min_value=0.0, value=0.0, step=0.01)
            with col_b:
                date = st.date_input("📅 Date (Optional)", value=datetime.now())
        
        with col2:
            st.markdown("### 📝 Quick Examples")
            st.markdown("*Click to try:*")
            
            examples = [
                ("🛒 Amazon", "AMAZON.COM*AB123"),
                ("☕ Starbucks", "STARBUCKS #456"),
                ("🚗 Uber", "UBER * TRIP ABC"),
                ("🎬 Netflix", "NETFLIX.COM"),
                ("⛽ Shell", "SHELL OIL 98765"),
                ("🍔 McDonald's", "MCDONALDS #789")
            ]
            
            for emoji_name, ex_text in examples:
                if st.button(emoji_name, key=f"ex_{ex_text}", width='stretch'):
                    st.session_state.selected_example = ex_text
                    st.rerun()
        
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            analyze_btn = st.button("🎯 Analyze Transaction", type="primary", width='stretch')
        
        if analyze_btn:
            if transaction_text:
                # Progress bar for effect
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Preprocessing text...")
                progress_bar.progress(25)
                time.sleep(0.2)
                
                cleaned = clean_text(transaction_text)
                
                status_text.text("🧠 Running AI analysis...")
                progress_bar.progress(60)
                time.sleep(0.3)
                
                result = model.predict_single(cleaned)
                
                status_text.text("✨ Generating insights...")
                progress_bar.progress(90)
                time.sleep(0.2)
                
                progress_bar.progress(100)
                time.sleep(0.2)
                progress_bar.empty()
                status_text.empty()
                
                # Display results with enhanced styling
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📁 Category", result['category'])
                
                with col2:
                    conf_pct = f"{result['confidence']:.1%}"
                    st.metric("🎲 Confidence", conf_pct)
                
                with col3:
                    if result['confidence'] > 0.85:
                        status = "🟢 High"
                        conf_class = "confidence-high"
                    elif result['confidence'] > 0.70:
                        status = "🟡 Medium"
                        conf_class = "confidence-medium"
                    else:
                        status = "🔴 Low"
                        conf_class = "confidence-low"
                    st.metric("📊 Status", status)
                
                with col4:
                    if amount > 0:
                        st.metric("💰 Amount", f"${amount:.2f}")
                    else:
                        st.metric("🔖 Processed", "✅")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed explanation
                col_exp1, col_exp2 = st.columns([1, 1])
                
                with col_exp1:
                    st.markdown("### 💡 AI Explanation")
                    st.info(result['explanation'])
                    
                    # Show cleaned text
                    with st.expander("🔍 View Processed Text"):
                        st.code(cleaned, language=None)
                
                with col_exp2:
                    st.markdown("### 🏆 Top 3 Predictions")
                    
                    for idx, (cat, prob) in enumerate(result['top_3_predictions'], 1):
                        medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
                        st.progress(prob, text=f"{medal} {cat}: {prob:.1%}")
                
                # Advanced visualization
                st.markdown("### 📊 Confidence Distribution")
                
                pred_data = pd.DataFrame(result['top_3_predictions'], columns=['Category', 'Probability'])
                
                fig = go.Figure()
                
                colors = ['#667eea', '#764ba2', '#f093fb']
                
                for idx, row in pred_data.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row['Probability']],
                        y=[row['Category']],
                        orientation='h',
                        name=row['Category'],
                        marker=dict(
                            color=colors[idx],
                            line=dict(color='white', width=2)
                        ),
                        text=f"{row['Probability']:.1%}",
                        textposition='auto',
                        hovertemplate=f"<b>{row['Category']}</b><br>Confidence: {row['Probability']:.2%}<extra></extra>"
                    ))
                
                fig.update_layout(
                    showlegend=False,
                    height=250,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(title="Confidence Score", range=[0, 1]),
                    yaxis=dict(title=""),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feedback section
                if result['confidence'] < 0.85:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ Low Confidence Detected")
                    st.write("This prediction has lower confidence. Please help improve the model by providing feedback!")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    with st.expander("✏️ Provide Correction", expanded=True):
                        correct_category = st.selectbox("Select Correct Category:", categories, key="correct_cat")
                        feedback_note = st.text_area("Additional Notes (Optional):", placeholder="Any specific reasons for the correction?")
                        
                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            if st.button("✅ Submit Feedback", width='stretch'):
                                feedback_system.add_feedback(
                                    description=cleaned,
                                    predicted_category=result['category'],
                                    true_category=correct_category,
                                    confidence=result['confidence'],
                                    amount=amount if amount > 0 else None
                                )
                                st.markdown('<div class="success-box">✅ Feedback recorded successfully! Thank you for helping improve the model.</div>', 
                                          unsafe_allow_html=True)
                                time.sleep(1)
                        
                        with col_f2:
                            if st.button("❌ Cancel", width='stretch'):
                                st.rerun()
                else:
                    st.balloons()
            else:
                st.warning("⚠️ Please enter a transaction description to analyze")
    
    with tab2:
        st.markdown("### 📁 Batch Transaction Processing")
        st.markdown("Upload a CSV file with transaction descriptions for bulk categorization")
        
        col_up1, col_up2 = st.columns([2, 1])
        
        with col_up1:
            uploaded_file = st.file_uploader(
                "Upload CSV File",
                type=['csv'],
                help="CSV must contain a 'description' column. Optional columns: amount, date",
                label_visibility="collapsed"
            )
        
        with col_up2:
            st.markdown("**📋 CSV Format:**")
            st.code("description,amount,date\nAMAZON.COM*123,45.99,2025-01-15", language="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            if 'description' not in df.columns:
                st.markdown('<div class="error-box">❌ <b>Error:</b> CSV must contain a <code>description</code> column</div>', 
                          unsafe_allow_html=True)
                return
            
            st.markdown(f'<div class="success-box">✅ Loaded <b>{len(df)}</b> transactions successfully</div>', 
                       unsafe_allow_html=True)
            
            # Preview
            with st.expander("👀 Preview Data", expanded=True):
                st.dataframe(df.head(10), width='stretch')
            
            col_batch1, col_batch2, col_batch3 = st.columns([1, 2, 1])
            with col_batch2:
                process_btn = st.button("🎯 Categorize All Transactions", type="primary", width='stretch')
            
            if process_btn:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Processing batch...")
                
                df['description_clean'] = df['description'].apply(clean_text)
                progress_bar.progress(30)
                
                status_text.text("🧠 Running AI predictions...")
                predictions, confidences = model.predict(df['description_clean'])
                progress_bar.progress(70)
                
                df['predicted_category'] = predictions
                df['confidence'] = confidences
                
                status_text.text("✨ Finalizing results...")
                progress_bar.progress(100)
                time.sleep(0.3)
                
                progress_bar.empty()
                status_text.empty()
                
                st.markdown('<div class="success-box">✅ <b>Processing Complete!</b> Categorized all transactions successfully.</div>', 
                          unsafe_allow_html=True)
                
                st.balloons()
                
                # Summary metrics
                st.markdown("### 📊 Processing Summary")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric("📝 Total Processed", len(df))
                
                with col_m2:
                    st.metric("🎯 Avg Confidence", f"{df['confidence'].mean():.1%}")
                
                with col_m3:
                    high_conf = (df['confidence'] >= 0.85).sum()
                    st.metric("🟢 High Confidence", high_conf)
                
                with col_m4:
                    low_conf = (df['confidence'] < 0.70).sum()
                    st.metric("🔴 Low Confidence", low_conf)
                
                # Results table
                st.markdown("### 📋 Categorized Transactions")
                
                # Add confidence indicator
                def style_confidence(val):
                    if val >= 0.85:
                        return 'background-color: #d4edda; color: #155724'
                    elif val >= 0.70:
                        return 'background-color: #fff3cd; color: #856404'
                    else:
                        return 'background-color: #f8d7da; color: #721c24'
                
                display_df = df[['description', 'predicted_category', 'confidence']].head(100)
                styled_df = display_df.style.applymap(style_confidence, subset=['confidence'])
                
                st.dataframe(styled_df, width='stretch', height=400)
                
                if len(df) > 100:
                    st.info(f"ℹ️ Showing first 100 of {len(df)} transactions. Download full results below.")
                
                # Download results
                col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
                
                with col_d2:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Results (CSV)",
                        data=csv,
                        file_name=f"categorized_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width='stretch'
                    )
                
                # Visualizations
                col_v1, col_v2 = st.columns(2)
                
                with col_v1:
                    st.markdown("#### 📊 Category Distribution")
                    cat_dist = df['predicted_category'].value_counts()
                    
                    fig = px.pie(
                        values=cat_dist.values,
                        names=cat_dist.index,
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_v2:
                    st.markdown("#### 🎲 Confidence Distribution")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df['confidence'],
                        nbinsx=20,
                        marker=dict(
                            color=df['confidence'],
                            colorscale='Viridis',
                            line=dict(color='white', width=1)
                        )
                    ))
                    fig.update_layout(
                        xaxis_title="Confidence Score",
                        yaxis_title="Count",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)


def show_performance_page(model):
    """Enhanced model performance metrics with advanced visualizations"""
    st.markdown("## 📊 Performance Analytics")
    st.markdown("Comprehensive model evaluation and performance metrics")
    
    if os.path.exists("evaluation_report.json"):
        with open("evaluation_report.json", 'r') as f:
            report = json.load(f)
        
        # Hero metrics with gradient cards
        st.markdown("### 🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            f1_score = report['macro_f1']
            delta = f"+{((f1_score - 0.90) / 0.90 * 100):.1f}% above target" if f1_score >= 0.90 else f"{((f1_score - 0.90) / 0.90 * 100):.1f}% below target"
            st.metric("🎯 Macro F1-Score", f"{f1_score:.4f}", delta=delta)
        
        with col2:
            accuracy = report['classification_report']['accuracy']
            st.metric("✅ Accuracy", f"{accuracy:.4f}")
        
        with col3:
            weighted_f1 = report['classification_report']['weighted avg']['f1-score']
            st.metric("⚖️ Weighted F1", f"{weighted_f1:.4f}")
        
        with col4:
            precision = report['classification_report']['weighted avg']['precision']
            st.metric("🎪 Precision", f"{precision:.4f}")
        
        st.markdown("---")
        
        # Performance gauge
        st.markdown("### 📈 F1-Score Performance Gauge")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=f1_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Macro F1-Score", 'font': {'size': 24}},
            delta={'reference': 0.90, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.70], 'color': '#f8d7da'},
                    {'range': [0.70, 0.85], 'color': '#fff3cd'},
                    {'range': [0.85, 0.90], 'color': '#d1ecf1'},
                    {'range': [0.90, 1], 'color': '#d4edda'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.90
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Per-category performance
        st.markdown("### � Per-Category Performance Analysis")
        
        per_class = pd.DataFrame(report['per_class_f1'].items(), columns=['Category', 'F1-Score'])
        per_class = per_class.sort_values('F1-Score', ascending=True)
        
        # Create a gradient color scale based on F1 scores
        colors = ['#eb3349' if x < 0.70 else '#f2994a' if x < 0.85 else '#56ab2f' 
                 for x in per_class['F1-Score']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=per_class['F1-Score'],
            y=per_class['Category'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f"{x:.3f}" for x in per_class['F1-Score']],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>F1-Score: %{x:.4f}<extra></extra>"
        ))
        
        fig.add_vline(x=0.90, line_dash="dash", line_color="red", line_width=3,
                     annotation_text="Target (0.90)", annotation_position="top right")
        
        fig.update_layout(
            xaxis_title="F1-Score",
            yaxis_title="Category",
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix with enhanced styling
        st.markdown("### 🔀 Confusion Matrix Heatmap")
        
        cm = pd.DataFrame(
            report['confusion_matrix'],
            index=list(report['per_class_f1'].keys()),
            columns=list(report['per_class_f1'].keys())
        )
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        
        fig.update_layout(
            height=600,
            xaxis_title="Predicted Category",
            yaxis_title="True Category"
        )
        
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification report details
        with st.expander("📋 Detailed Classification Report"):
            report_df = pd.DataFrame(report['classification_report']).T
            report_df = report_df[report_df.index != 'accuracy']
            st.dataframe(report_df.style.format("{:.4f}"), width='stretch')
        
        # Model comparison (if multiple models)
        st.markdown("### 🏆 Model Comparison")
        
        comparison_data = {
            'Metric': ['F1-Score', 'Accuracy', 'Precision', 'Recall'],
            'Current Model': [
                f1_score,
                accuracy,
                precision,
                report['classification_report']['weighted avg']['recall']
            ],
            'Target': [0.90, 0.90, 0.90, 0.90],
            'Baseline': [0.67, 0.70, 0.68, 0.66]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        
        metrics = comp_df['Metric']
        
        fig.add_trace(go.Scatter(
            x=metrics, y=comp_df['Current Model'],
            mode='lines+markers',
            name='Current Model',
            line=dict(color='#667eea', width=3),
            marker=dict(size=12)
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics, y=comp_df['Target'],
            mode='lines+markers',
            name='Target',
            line=dict(color='#56ab2f', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics, y=comp_df['Baseline'],
            mode='lines+markers',
            name='Baseline',
            line=dict(color='#eb3349', width=2, dash='dot'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.markdown('<div class="warning-box">⚠️ <b>No evaluation report found.</b> Run <code>python evaluate_model.py</code> to generate performance metrics.</div>', 
                   unsafe_allow_html=True)


def show_category_management_page(config):
    """Enhanced category configuration management"""
    st.markdown("## 🎨 Category Designer")
    st.markdown("Manage and customize transaction categories")
    
    st.markdown('<div class="success-box">� <b>Pro Tip:</b> Categories are defined in <code>config/categories.yaml</code>. Edit keywords and retrain for customization!</div>', 
               unsafe_allow_html=True)
    
    # Summary stats
    total_categories = len(config['categories'])
    total_keywords = sum(len(cat['keywords']) for cat in config['categories'])
    avg_keywords = total_keywords / total_categories if total_categories > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 Total Categories", total_categories)
    with col2:
        st.metric("🔑 Total Keywords", total_keywords)
    with col3:
        st.metric("📊 Avg Keywords/Category", f"{avg_keywords:.1f}")
    
    st.markdown("---")
    
    # Search and filter
    search_term = st.text_input("🔍 Search Categories", placeholder="Enter category name or keyword...")
    
    # Display categories in a grid
    st.markdown("### 📋 Category Overview")
    
    cols = st.columns(2)
    
    for idx, cat_info in enumerate(config['categories']):
        cat_name = cat_info['name']
        
        # Filter by search
        if search_term and search_term.lower() not in cat_name.lower() and \
           not any(search_term.lower() in kw.lower() for kw in cat_info['keywords']):
            continue
        
        with cols[idx % 2]:
            with st.expander(f"📁 **{cat_name}** ({len(cat_info['keywords'])} keywords)", expanded=False):
                # Category info
                st.markdown(f"**Description:** {cat_info.get('description', 'No description available')}")
                
                # Keywords as pills
                st.markdown("**Keywords:**")
                keyword_html = " ".join([f'<span class="category-pill">{kw}</span>' for kw in cat_info['keywords']])
                st.markdown(keyword_html, unsafe_allow_html=True)
                
                # Show keyword count
                st.caption(f"Total: {len(cat_info['keywords'])} keywords")
    
    st.markdown("---")
    
    # Add new category section
    st.markdown("### ➕ Add New Category")
    
    col_add1, col_add2 = st.columns(2)
    
    with col_add1:
        new_cat_name = st.text_input("Category Name", placeholder="e.g., Subscriptions")
        new_cat_desc = st.text_area("Description", placeholder="e.g., Monthly subscription services")
    
    with col_add2:
        new_cat_keywords = st.text_area(
            "Keywords (comma-separated)",
            placeholder="netflix, spotify, youtube, disney",
            help="Enter keywords separated by commas"
        )
        
        st.markdown("**Example YAML:**")
        st.code(f"""- name: "{new_cat_name or 'Your Category'}"
  description: "{new_cat_desc or 'Category description'}"
  keywords:
    - keyword1
    - keyword2""", language="yaml")
    
    if st.button("ℹ️ How to Add", width='stretch'):
        st.info("""
**Steps to add a new category:**

1. Open `config/categories.yaml` file
2. Add your new category following the YAML format
3. Save the file
4. Run `python train_model.py` to retrain the model
5. Your new category will be available!
        """)
    
    st.markdown("---")
    
    # Category statistics visualization
    st.markdown("### 📊 Category Statistics")
    
    cat_stats = pd.DataFrame([
        {
            'Category': cat['name'],
            'Keywords': len(cat['keywords']),
            'Has Description': 'Yes' if cat.get('description') else 'No'
        }
        for cat in config['categories']
    ])
    
    fig = px.bar(
        cat_stats.sort_values('Keywords', ascending=True),
        x='Keywords',
        y='Category',
        orientation='h',
        color='Keywords',
        color_continuous_scale='Purples',
        title="Keywords per Category"
    )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_feedback_page(feedback_system):
    """Enhanced feedback system interface"""
    st.markdown("## 💬 Feedback Hub")
    st.markdown("Track and manage user feedback for continuous model improvement")
    
    stats = feedback_system.get_feedback_stats()
    
    # Hero statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Feedback", stats['total_feedback'])
    
    with col2:
        st.metric("✏️ Corrections", stats['corrections'])
    
    with col3:
        accuracy_pct = f"{stats['accuracy_on_feedback']:.1%}"
        st.metric("🎯 Feedback Accuracy", accuracy_pct)
    
    with col4:
        if stats['total_feedback'] > 0:
            correction_rate = (stats['corrections'] / stats['total_feedback']) * 100
            st.metric("📊 Correction Rate", f"{correction_rate:.1f}%")
        else:
            st.metric("📊 Correction Rate", "N/A")
    
    if stats['total_feedback'] > 0:
        st.markdown("---")
        
        # Feedback insights
        st.markdown("### 📈 Feedback Insights")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Feedback timeline (mock data - could be enhanced with real dates)
            st.markdown("#### � Feedback Over Time")
            feedback_df = feedback_system.get_feedback_data()
            
            if 'timestamp' in feedback_df.columns:
                feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
                timeline = feedback_df.groupby('date').size().reset_index(name='count')
                
                fig = px.line(
                    timeline,
                    x='date',
                    y='count',
                    markers=True,
                    title="Daily Feedback Count"
                )
                fig.update_traces(line_color='#667eea', line_width=3)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Timestamp data not available for timeline visualization")
        
        with col_chart2:
            # Confidence distribution
            st.markdown("#### 🎲 Confidence Distribution")
            
            if 'confidence' in feedback_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=feedback_df['confidence'],
                    nbinsx=20,
                    marker=dict(
                        color='#764ba2',
                        line=dict(color='white', width=1)
                    )
                ))
                fig.update_layout(
                    xaxis_title="Confidence Score",
                    yaxis_title="Count",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Confidence data not available")
        
        # Recent feedback table
        st.markdown("### 📋 Recent Feedback Entries")
        
        display_cols = ['description', 'predicted_category', 'true_category', 'confidence']
        available_cols = [col for col in display_cols if col in feedback_df.columns]
        
        if available_cols:
            recent_feedback = feedback_df[available_cols].tail(50)
            st.dataframe(recent_feedback, width='stretch', height=400)
        else:
            st.dataframe(feedback_df.tail(50), width='stretch', height=400)
        
        # Export functionality
        col_exp1, col_exp2, col_exp3 = st.columns([1, 2, 1])
        
        with col_exp2:
            if st.button("📥 Export All Feedback Data", width='stretch'):
                export_file = f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                feedback_system.export_feedback_for_review(export_file)
                st.markdown(f'<div class="success-box">✅ Exported to <code>{export_file}</code></div>', 
                          unsafe_allow_html=True)
        
        # Confused pairs analysis
        if stats['most_confused_pairs']:
            st.markdown("---")
            st.markdown("### 🔀 Most Confused Category Pairs")
            st.caption("These category pairs are most frequently confused by the model")
            
            confused_data = pd.DataFrame(
                stats['most_confused_pairs'],
                columns=['Predicted', 'Actual', 'Count']
            )
            
            # Visualization
            fig = px.bar(
                confused_data.head(10),
                x='Count',
                y=[f"{row['Predicted']} → {row['Actual']}" for _, row in confused_data.head(10).iterrows()],
                orientation='h',
                color='Count',
                color_continuous_scale='Reds',
                title="Top 10 Confused Pairs"
            )
            fig.update_layout(yaxis_title="Category Pair", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table view
            with st.expander("📊 View All Confused Pairs"):
                st.dataframe(confused_data, width='stretch')
        
        # Retraining option
        st.markdown("---")
        st.markdown("### 🔄 Model Retraining")
        
        if stats['total_feedback'] >= 50:
            st.markdown('<div class="success-box">✅ You have enough feedback data for retraining! (Minimum: 50 samples)</div>', 
                       unsafe_allow_html=True)
            
            if st.button("🚀 Retrain Model with Feedback", type="primary"):
                with st.spinner("Retraining model with feedback..."):
                    # This would require access to original training data
                    st.info("⚠️ To retrain with feedback, run the retraining script manually with access to original training data.")
        else:
            remaining = 50 - stats['total_feedback']
            st.markdown(f'<div class="warning-box">⏳ Need {remaining} more feedback samples for retraining (Current: {stats["total_feedback"]}/50)</div>', 
                       unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="warning-box">📭 <b>No feedback collected yet.</b><br>Use the Smart Categorizer page to provide feedback on predictions with low confidence.</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("### 💡 Why Provide Feedback?")
        
        col_why1, col_why2, col_why3 = st.columns(3)
        
        with col_why1:
            st.markdown("""
            **🎯 Improve Accuracy**
            
            Your corrections help the model learn from mistakes and improve future predictions.
            """)
        
        with col_why2:
            st.markdown("""
            **📊 Track Performance**
            
            Feedback data helps identify problematic categories and patterns.
            """)
        
        with col_why3:
            st.markdown("""
            **🔄 Active Learning**
            
            The model can be retrained with feedback for continuous improvement.
            """)


def show_analytics_dashboard(model):
    """Enhanced analytics dashboard with rich visualizations"""
    st.markdown("## 📈 Insights Dashboard")
    st.markdown("Comprehensive data analytics and transaction insights")
    
    # Load test data if available
    if os.path.exists("data/test_transactions.csv"):
        df = pd.read_csv("data/test_transactions.csv")
        
        # Overview metrics
        st.markdown("### 📊 Dataset Overview")
        
        col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
        
        with col_ov1:
            st.metric("📝 Total Transactions", len(df))
        
        with col_ov2:
            if 'amount' in df.columns:
                total_amount = df['amount'].sum()
                st.metric("💰 Total Amount", f"${total_amount:,.2f}")
            else:
                st.metric("💰 Total Amount", "N/A")
        
        with col_ov3:
            unique_cats = df['category'].nunique()
            st.metric("📁 Unique Categories", unique_cats)
        
        with col_ov4:
            if 'amount' in df.columns:
                avg_amount = df['amount'].mean()
                st.metric("📊 Avg Amount", f"${avg_amount:.2f}")
            else:
                st.metric("📊 Avg Amount", "N/A")
        
        st.markdown("---")
        
        # Main visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("#### 📊 Category Distribution")
            cat_dist = df['category'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=cat_dist.index,
                values=cat_dist.values,
                hole=.4,
                marker=dict(
                    colors=px.colors.qualitative.Set3,
                    line=dict(color='white', width=2)
                ),
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
            )])
            
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False,
                annotations=[dict(text='Categories', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_viz2:
            st.markdown("#### 💰 Amount Distribution by Category")
            if 'amount' in df.columns:
                fig = go.Figure()
                
                for cat in df['category'].unique():
                    cat_data = df[df['category'] == cat]['amount']
                    fig.add_trace(go.Box(
                        y=cat_data,
                        name=cat,
                        boxmean='sd'
                    ))
                
                fig.update_layout(
                    yaxis_title="Amount ($)",
                    height=400,
                    showlegend=True,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Amount data not available in dataset")
        
        # Timeline analysis
        if 'date' in df.columns:
            st.markdown("---")
            st.markdown("### 📅 Temporal Analysis")
            
            df['date'] = pd.to_datetime(df['date'])
            
            tab_time1, tab_time2 = st.tabs(["📈 Time Series", "🔥 Heatmap"])
            
            with tab_time1:
                # Monthly transactions
                df['month'] = df['date'].dt.to_period('M').astype(str)
                timeline = df.groupby(['month', 'category']).size().reset_index(name='count')
                
                fig = px.area(
                    timeline,
                    x='month',
                    y='count',
                    color='category',
                    title="Monthly Transaction Trends by Category"
                )
                
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Transaction Count",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab_time2:
                # Day of week heatmap
                df['day_of_week'] = df['date'].dt.day_name()
                df['hour'] = df['date'].dt.hour if df['date'].dt.hour.notna().any() else 12
                
                # Create heatmap data
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_data = df.groupby(['day_of_week', df['date'].dt.hour]).size().unstack(fill_value=0)
                heatmap_data = heatmap_data.reindex(day_order)
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='Viridis',
                    hovertemplate="Day: %{y}<br>Hour: %{x}<br>Count: %{z}<extra></extra>"
                ))
                
                fig.update_layout(
                    title="Transaction Heatmap by Day and Hour",
                    xaxis_title="Hour of Day",
                    yaxis_title="Day of Week",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Advanced analytics
        st.markdown("---")
        st.markdown("### 🔬 Advanced Analytics")
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            st.markdown("#### 📊 Category Statistics")
            
            cat_stats = df.groupby('category').agg({
                'description': 'count',
                'amount': ['mean', 'sum'] if 'amount' in df.columns else 'count'
            }).reset_index()
            
            if 'amount' in df.columns:
                cat_stats.columns = ['Category', 'Count', 'Avg Amount', 'Total Amount']
                cat_stats = cat_stats.sort_values('Total Amount', ascending=False)
            else:
                cat_stats.columns = ['Category', 'Count']
                cat_stats = cat_stats.sort_values('Count', ascending=False)
            
            st.dataframe(cat_stats, width='stretch', height=400)
        
        with col_adv2:
            st.markdown("#### 🏆 Top Merchants")
            
            # Extract merchant patterns (simplified)
            top_merchants = df['description'].value_counts().head(10)
            
            fig = px.bar(
                x=top_merchants.values,
                y=top_merchants.index,
                orientation='h',
                color=top_merchants.values,
                color_continuous_scale='Teal',
                labels={'x': 'Count', 'y': 'Merchant'}
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Download processed data
        st.markdown("---")
        
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        
        with col_dl2:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Analytics Data (CSV)",
                data=csv_data,
                file_name=f"analytics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
    
    else:
        st.markdown('<div class="warning-box">⚠️ <b>No data available for analytics.</b><br>Generate test data by running <code>python -m src.data_generator</code></div>', 
                   unsafe_allow_html=True)
        
        st.markdown("### 📊 Sample Analytics Features")
        
        col_feat1, col_feat2, col_feat3 = st.columns(3)
        
        with col_feat1:
            st.markdown("""
            **📈 Time Series Analysis**
            
            - Monthly trends
            - Seasonal patterns
            - Day/hour heatmaps
            """)
        
        with col_feat2:
            st.markdown("""
            **💰 Financial Insights**
            
            - Amount distributions
            - Category spending
            - Top merchants
            """)
        
        with col_feat3:
            st.markdown("""
            **📊 Statistical Analysis**
            
            - Category distributions
            - Transaction patterns
            - Data exports
            """)


if __name__ == "__main__":
    main()
