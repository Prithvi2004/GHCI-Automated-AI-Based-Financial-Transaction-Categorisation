"""
Streamlit Web UI for Transaction Categorization
Interactive web interface with visualization and configuration
"""

import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from src.preprocess import clean_text
from src.feedback import FeedbackSystem
from src.explainability import TransactionExplainer
import yaml


# Page configuration
st.set_page_config(
    page_title="Transaction Categorization System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
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
    st.markdown('<div class="main-header">💰 AI-Powered Transaction Categorization</div>', 
                unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model not found. Please train the model first by running `python train_model.py`")
        return
    
    config = load_config()
    categories = model.get_categories()
    feedback_system = FeedbackSystem()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        page = st.radio("Navigation", [
            "🔮 Categorize Transactions",
            "📊 Model Performance",
            "⚙️ Category Management",
            "💬 Feedback System",
            "📈 Analytics Dashboard"
        ])
        
        st.markdown("---")
        st.info(f"**Model Status:** ✅ Ready\n\n**Categories:** {len(categories)}")
    
    # Main content based on page selection
    if page == "🔮 Categorize Transactions":
        show_categorization_page(model, categories, feedback_system)
    
    elif page == "📊 Model Performance":
        show_performance_page(model)
    
    elif page == "⚙️ Category Management":
        show_category_management_page(config)
    
    elif page == "💬 Feedback System":
        show_feedback_page(feedback_system)
    
    elif page == "📈 Analytics Dashboard":
        show_analytics_dashboard(model)


def show_categorization_page(model, categories, feedback_system):
    """Transaction categorization interface"""
    st.header("🔮 Categorize Transactions")
    
    # Input options
    input_method = st.radio("Input Method:", ["Single Transaction", "Batch Upload"])
    
    if input_method == "Single Transaction":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            transaction_text = st.text_input(
                "Enter Transaction Description:",
                placeholder="e.g., STARBUCKS STORE #12345",
                help="Enter the merchant name or transaction description"
            )
            
            amount = st.number_input("Amount (Optional):", min_value=0.0, value=0.0, step=0.01)
        
        with col2:
            st.markdown("### Examples")
            examples = [
                "AMAZON.COM*123",
                "STARBUCKS #456",
                "UBER * TRIP",
                "NETFLIX.COM"
            ]
            
            for ex in examples:
                if st.button(ex, key=f"ex_{ex}"):
                    transaction_text = ex
        
        if st.button("🎯 Categorize", type="primary"):
            if transaction_text:
                with st.spinner("Analyzing..."):
                    cleaned = clean_text(transaction_text)
                    result = model.predict_single(cleaned)
                    
                    # Display results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Category", result['category'])
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with col3:
                        conf_color = "🟢" if result['confidence'] > 0.85 else "🟡" if result['confidence'] > 0.7 else "🔴"
                        st.metric("Status", f"{conf_color} {'High' if result['confidence'] > 0.85 else 'Medium' if result['confidence'] > 0.7 else 'Low'}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Explanation
                    st.subheader("💡 Explanation")
                    st.info(result['explanation'])
                    
                    # Top predictions visualization
                    st.subheader("🏆 Top Predictions")
                    
                    pred_data = pd.DataFrame(result['top_3_predictions'], columns=['Category', 'Probability'])
                    fig = px.bar(pred_data, x='Probability', y='Category', orientation='h',
                                color='Probability', color_continuous_scale='Blues')
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feedback option
                    if result['confidence'] < 0.85:
                        st.warning("⚠️ Low confidence prediction. Please provide feedback if incorrect.")
                        
                        with st.expander("✏️ Provide Feedback"):
                            correct_category = st.selectbox("Correct Category:", categories)
                            if st.button("Submit Feedback"):
                                feedback_system.add_feedback(
                                    description=cleaned,
                                    predicted_category=result['category'],
                                    true_category=correct_category,
                                    confidence=result['confidence'],
                                    amount=amount if amount > 0 else None
                                )
                                st.success("✅ Feedback recorded. Thank you!")
            else:
                st.warning("Please enter a transaction description")
    
    else:  # Batch Upload
        st.subheader("📁 Batch Processing")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'],
                                        help="CSV should have a 'description' column")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            if 'description' not in df.columns:
                st.error("CSV must contain a 'description' column")
                return
            
            st.info(f"Loaded {len(df)} transactions")
            
            if st.button("🎯 Categorize All"):
                with st.spinner("Processing batch..."):
                    df['description_clean'] = df['description'].apply(clean_text)
                    predictions, confidences = model.predict(df['description_clean'])
                    
                    df['predicted_category'] = predictions
                    df['confidence'] = confidences
                    
                    st.success(f"✅ Categorized {len(df)} transactions")
                    
                    # Display results
                    st.dataframe(df[['description', 'predicted_category', 'confidence']].head(100))
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="categorized_transactions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Processed", len(df))
                    with col2:
                        st.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
                    with col3:
                        low_conf = (df['confidence'] < 0.85).sum()
                        st.metric("Low Confidence", low_conf)
                    
                    # Category distribution
                    st.subheader("📊 Category Distribution")
                    cat_dist = df['predicted_category'].value_counts()
                    fig = px.pie(values=cat_dist.values, names=cat_dist.index)
                    st.plotly_chart(fig)


def show_performance_page(model):
    """Model performance metrics"""
    st.header("📊 Model Performance")
    
    if os.path.exists("evaluation_report.json"):
        import json
        with open("evaluation_report.json", 'r') as f:
            report = json.load(f)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Macro F1-Score", f"{report['macro_f1']:.3f}",
                     delta="Target: 0.90", delta_color="normal" if report['macro_f1'] >= 0.90 else "inverse")
        with col2:
            st.metric("Accuracy", f"{report['classification_report']['accuracy']:.3f}")
        with col3:
            st.metric("Weighted F1", f"{report['classification_report']['weighted avg']['f1-score']:.3f}")
        
        # Per-category performance
        st.subheader("📈 Per-Category Performance")
        
        per_class = pd.DataFrame(report['per_class_f1'].items(), columns=['Category', 'F1-Score'])
        per_class = per_class.sort_values('F1-Score', ascending=True)
        
        fig = px.bar(per_class, x='F1-Score', y='Category', orientation='h',
                    color='F1-Score', color_continuous_scale='RdYlGn',
                    range_color=[0, 1])
        fig.add_vline(x=0.90, line_dash="dash", line_color="red", 
                     annotation_text="Target (0.90)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        st.subheader("🔀 Confusion Matrix")
        cm = pd.DataFrame(report['confusion_matrix'], 
                         index=list(report['per_class_f1'].keys()),
                         columns=list(report['per_class_f1'].keys()))
        
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       color_continuous_scale='Blues')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No evaluation report found. Run `python evaluate_model.py` first.")


def show_category_management_page(config):
    """Category configuration management"""
    st.header("⚙️ Category Management")
    
    st.info("📝 Categories are defined in `config/categories.yaml`. Edit the file to customize.")
    
    for cat_info in config['categories']:
        with st.expander(f"📁 {cat_info['name']}"):
            st.write(f"**Description:** {cat_info.get('description', 'N/A')}")
            st.write(f"**Keywords ({len(cat_info['keywords'])}):**")
            st.write(", ".join(cat_info['keywords']))
    
    st.markdown("---")
    st.markdown("### ➕ Add New Category")
    st.info("To add a new category, edit `config/categories.yaml` and retrain the model.")


def show_feedback_page(feedback_system):
    """Feedback system interface"""
    st.header("💬 Feedback System")
    
    stats = feedback_system.get_feedback_stats()
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Feedback", stats['total_feedback'])
    with col2:
        st.metric("Corrections", stats['corrections'])
    with col3:
        st.metric("Accuracy", f"{stats['accuracy_on_feedback']:.1%}")
    
    if stats['total_feedback'] > 0:
        # Feedback data
        st.subheader("📋 Recent Feedback")
        feedback_df = feedback_system.get_feedback_data()
        st.dataframe(feedback_df.tail(50))
        
        # Export
        if st.button("📥 Export All Feedback"):
            feedback_system.export_feedback_for_review("feedback_export.csv")
            st.success("✅ Exported to feedback_export.csv")
        
        # Confused pairs
        if stats['most_confused_pairs']:
            st.subheader("🔀 Most Confused Category Pairs")
            confused_data = pd.DataFrame(
                stats['most_confused_pairs'],
                columns=['Predicted', 'Actual', 'Count']
            )
            st.table(confused_data)
    else:
        st.info("No feedback collected yet. Use the categorization page to provide feedback.")


def show_analytics_dashboard(model):
    """Analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    # Load test data if available
    if os.path.exists("data/test_transactions.csv"):
        df = pd.read_csv("data/test_transactions.csv")
        
        # Category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dataset Category Distribution")
            cat_dist = df['category'].value_counts()
            fig = px.pie(values=cat_dist.values, names=cat_dist.index,
                        title="Category Distribution")
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("💰 Amount Distribution by Category")
            if 'amount' in df.columns:
                fig = px.box(df, x='category', y='amount',
                            title="Transaction Amounts by Category")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig)
        
        # Timeline if date available
        if 'date' in df.columns:
            st.subheader("📅 Transactions Over Time")
            df['date'] = pd.to_datetime(df['date'])
            timeline = df.groupby([df['date'].dt.to_period('M'), 'category']).size().reset_index(name='count')
            timeline['date'] = timeline['date'].astype(str)
            
            fig = px.line(timeline, x='date', y='count', color='category',
                         title="Monthly Transaction Count by Category")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No test data available for analytics.")


if __name__ == "__main__":
    main()
