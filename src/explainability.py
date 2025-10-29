"""
Explainability Module for Transaction Categorization
Implements LIME-based explanations and feature attribution
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


class TransactionExplainer:
    """
    Provides explainability for transaction categorization predictions
    Uses LIME (Local Interpretable Model-agnostic Explanations) when available
    """
    
    def __init__(self, model):
        """
        Initialize explainer with trained model
        
        Args:
            model: Trained TransactionCategorizer instance
        """
        self.model = model
        self.lime_explainer = None
        
        # Try to import LIME
        try:
            from lime.lime_text import LimeTextExplainer
            self.lime_available = True
            self.lime_explainer = LimeTextExplainer(class_names=model.get_categories())
        except ImportError:
            self.lime_available = False
            print("⚠️ LIME not installed. Install with: pip install lime")
    
    def explain_prediction(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction
        
        Args:
            text: Transaction description
            num_features: Number of features to show in explanation
            
        Returns:
            Dictionary with explanation details
        """
        # Get prediction
        result = self.model.predict_single(text)
        
        explanation = {
            'transaction': text,
            'predicted_category': result['category'],
            'confidence': result['confidence'],
            'top_3_predictions': result['top_3_predictions'],
            'keyword_explanation': result['explanation'],
            'lime_explanation': None
        }
        
        # Add LIME explanation if available
        if self.lime_available and self.lime_explainer:
            try:
                lime_exp = self._get_lime_explanation(text, num_features)
                explanation['lime_explanation'] = lime_exp
            except Exception as e:
                explanation['lime_explanation'] = f"LIME explanation failed: {str(e)}"
        
        return explanation
    
    def _get_lime_explanation(self, text: str, num_features: int) -> Dict[str, Any]:
        """
        Generate LIME explanation for prediction
        
        Args:
            text: Transaction description
            num_features: Number of features to include
            
        Returns:
            LIME explanation details
        """
        # Create prediction function for LIME
        def predict_proba(texts):
            _, proba = self.model.predict(texts, return_proba=True)
            return proba
        
        # Generate LIME explanation
        exp = self.lime_explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features,
            num_samples=500
        )
        
        # Get predicted class index
        pred_class = self.model.predict([text])[0][0]
        class_idx = self.model.get_categories().index(pred_class)
        
        # Extract feature weights
        feature_weights = exp.as_list(label=class_idx)
        
        return {
            'important_words': feature_weights,
            'prediction_probability': exp.predict_proba[class_idx],
            'local_prediction': exp.local_pred[0] if hasattr(exp, 'local_pred') else None
        }
    
    def explain_batch(self, texts: List[str], num_features: int = 5) -> pd.DataFrame:
        """
        Generate explanations for multiple transactions
        
        Args:
            texts: List of transaction descriptions
            num_features: Number of features per explanation
            
        Returns:
            DataFrame with predictions and explanations
        """
        explanations = []
        
        for text in texts:
            exp = self.explain_prediction(text, num_features)
            
            # Extract LIME top words if available
            top_words = ""
            if exp['lime_explanation'] and isinstance(exp['lime_explanation'], dict):
                important = exp['lime_explanation'].get('important_words', [])
                top_words = ', '.join([f"{word}({weight:.2f})" for word, weight in important[:3]])
            
            explanations.append({
                'transaction': text,
                'category': exp['predicted_category'],
                'confidence': exp['confidence'],
                'keyword_match': exp['keyword_explanation'],
                'lime_features': top_words
            })
        
        return pd.DataFrame(explanations)
    
    def get_global_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance from model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = self.model.get_feature_importance()
        
        if importance is None:
            return {}
        
        # For now, return top features with generic names
        # In full implementation, would map to actual feature names
        top_n = min(20, len(importance))
        top_indices = np.argsort(importance)[-top_n:][::-1]
        
        return {
            f'feature_{i}': importance[i]
            for i in top_indices
        }
    
    def visualize_explanation(self, text: str, save_path: str = None) -> str:
        """
        Create visualization of explanation
        
        Args:
            text: Transaction description
            save_path: Path to save visualization (optional)
            
        Returns:
            HTML string or path to saved file
        """
        if not self.lime_available:
            return "LIME not available for visualization"
        
        # Generate explanation
        exp_dict = self.explain_prediction(text, num_features=10)
        
        # Create HTML report
        html = f"""
        <html>
        <head><title>Transaction Explanation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #4CAF50; color: white; padding: 15px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
            .feature {{ display: inline-block; margin: 5px; padding: 8px; background: #e3f2fd; border-radius: 3px; }}
            .positive {{ background: #c8e6c9; }}
            .negative {{ background: #ffcdd2; }}
            table {{ width: 100%; border-collapse: collapse; }}
            td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        </style>
        </head>
        <body>
            <div class="header">
                <h1>Transaction Categorization Explanation</h1>
            </div>
            
            <div class="section">
                <h2>Transaction: "{text}"</h2>
                <p><strong>Predicted Category:</strong> {exp_dict['predicted_category']}</p>
                <p><strong>Confidence:</strong> {exp_dict['confidence']:.2%}</p>
            </div>
            
            <div class="section">
                <h3>Top Predictions</h3>
                <table>
                    <tr><th>Category</th><th>Probability</th></tr>
        """
        
        for cat, prob in exp_dict['top_3_predictions']:
            html += f"<tr><td>{cat}</td><td>{prob:.2%}</td></tr>"
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h3>Keyword Explanation</h3>
                <p>{}</p>
            </div>
        """.format(exp_dict['keyword_explanation'])
        
        if exp_dict['lime_explanation'] and isinstance(exp_dict['lime_explanation'], dict):
            html += """
            <div class="section">
                <h3>LIME Feature Attribution</h3>
                <p>Words contributing to the prediction:</p>
            """
            
            for word, weight in exp_dict['lime_explanation']['important_words']:
                cls = 'positive' if weight > 0 else 'negative'
                html += f'<span class="feature {cls}">{word}: {weight:.3f}</span>'
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html)
            return save_path
        
        return html


def demonstrate_explainability(model, sample_transactions: List[str]):
    """
    Demonstrate explainability features with sample transactions
    
    Args:
        model: Trained TransactionCategorizer
        sample_transactions: List of sample transaction descriptions
    """
    print("\n" + "="*60)
    print("🔍 EXPLAINABILITY DEMONSTRATION")
    print("="*60)
    
    explainer = TransactionExplainer(model)
    
    for i, text in enumerate(sample_transactions, 1):
        print(f"\n📝 Transaction {i}: {text}")
        print("-" * 60)
        
        explanation = explainer.explain_prediction(text, num_features=5)
        
        print(f"🎯 Predicted: {explanation['predicted_category']}")
        print(f"📊 Confidence: {explanation['confidence']:.2%}")
        print(f"💡 Explanation: {explanation['keyword_explanation']}")
        
        print("\n🏆 Top 3 Predictions:")
        for cat, prob in explanation['top_3_predictions']:
            print(f"  • {cat}: {prob:.2%}")
        
        if explanation['lime_explanation'] and isinstance(explanation['lime_explanation'], dict):
            print("\n🔬 LIME Analysis - Important Words:")
            for word, weight in explanation['lime_explanation']['important_words'][:5]:
                indicator = "+" if weight > 0 else "-"
                print(f"  {indicator} {word}: {abs(weight):.3f}")
    
    print("\n" + "="*60)
