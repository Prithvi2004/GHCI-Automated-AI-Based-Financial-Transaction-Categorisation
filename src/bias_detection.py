"""
Bias Detection and Mitigation Module
Analyzes model fairness across different transaction types and amounts
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns


class BiasAnalyzer:
    """
    Analyzes potential biases in transaction categorization model
    """
    
    def __init__(self, model):
        """
        Initialize bias analyzer
        
        Args:
            model: Trained TransactionCategorizer instance
        """
        self.model = model
    
    def analyze_amount_bias(self, df: pd.DataFrame, amount_col='amount', 
                           true_col='category', pred_col='predicted') -> Dict[str, Any]:
        """
        Analyze if model performance varies by transaction amount
        
        Args:
            df: DataFrame with transactions, amounts, true and predicted categories
            amount_col: Column name for transaction amounts
            true_col: Column name for true categories
            pred_col: Column name for predicted categories
            
        Returns:
            Analysis results dictionary
        """
        # Define amount bins
        amount_bins = [0, 20, 50, 100, 200, float('inf')]
        amount_labels = ['$0-20', '$20-50', '$50-100', '$100-200', '$200+']
        
        df['amount_bin'] = pd.cut(df[amount_col], bins=amount_bins, labels=amount_labels)
        
        # Calculate metrics per bin
        results = {}
        for bin_label in amount_labels:
            bin_data = df[df['amount_bin'] == bin_label]
            
            if len(bin_data) == 0:
                continue
            
            results[bin_label] = {
                'count': len(bin_data),
                'accuracy': accuracy_score(bin_data[true_col], bin_data[pred_col]),
                'f1_macro': f1_score(bin_data[true_col], bin_data[pred_col], average='macro', zero_division=0),
                'precision': precision_score(bin_data[true_col], bin_data[pred_col], average='macro', zero_division=0),
                'recall': recall_score(bin_data[true_col], bin_data[pred_col], average='macro', zero_division=0)
            }
        
        # Calculate variance in performance
        f1_scores = [r['f1_macro'] for r in results.values()]
        bias_score = np.std(f1_scores) if len(f1_scores) > 0 else 0
        
        return {
            'by_amount_range': results,
            'bias_score': bias_score,
            'interpretation': self._interpret_bias(bias_score)
        }
    
    def analyze_category_balance(self, y_true, y_pred) -> Dict[str, Any]:
        """
        Analyze if model performs differently across categories
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Category-level performance metrics
        """
        from sklearn.metrics import classification_report
        
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Extract per-category metrics
        categories = self.model.get_categories()
        category_metrics = {}
        
        for cat in categories:
            if cat in report:
                category_metrics[cat] = {
                    'f1_score': report[cat]['f1-score'],
                    'precision': report[cat]['precision'],
                    'recall': report[cat]['recall'],
                    'support': report[cat]['support']
                }
        
        # Calculate performance variance
        f1_scores = [m['f1_score'] for m in category_metrics.values()]
        variance = np.var(f1_scores) if len(f1_scores) > 0 else 0
        
        return {
            'category_metrics': category_metrics,
            'f1_variance': variance,
            'min_f1': min(f1_scores) if f1_scores else 0,
            'max_f1': max(f1_scores) if f1_scores else 0,
            'interpretation': self._interpret_category_balance(variance)
        }
    
    def analyze_merchant_type_bias(self, df: pd.DataFrame, 
                                   desc_col='description', 
                                   true_col='category',
                                   pred_col='predicted') -> Dict[str, Any]:
        """
        Analyze bias based on merchant name characteristics
        
        Args:
            df: DataFrame with transactions
            desc_col: Column name for descriptions
            true_col: True category column
            pred_col: Predicted category column
            
        Returns:
            Bias analysis results
        """
        # Categorize merchants by characteristics
        df['has_number'] = df[desc_col].str.contains(r'\d', regex=True)
        df['has_special_char'] = df[desc_col].str.contains(r'[#*]', regex=True)
        df['length_category'] = pd.cut(
            df[desc_col].str.len(), 
            bins=[0, 15, 25, 100], 
            labels=['short', 'medium', 'long']
        )
        
        results = {}
        
        # Analyze by each characteristic
        for char in ['has_number', 'has_special_char', 'length_category']:
            char_results = {}
            
            for value in df[char].unique():
                subset = df[df[char] == value]
                if len(subset) > 0:
                    char_results[str(value)] = {
                        'count': len(subset),
                        'accuracy': accuracy_score(subset[true_col], subset[pred_col]),
                        'f1_macro': f1_score(subset[true_col], subset[pred_col], 
                                            average='macro', zero_division=0)
                    }
            
            results[char] = char_results
        
        return results
    
    def generate_bias_report(self, df: pd.DataFrame, save_path: str = None) -> str:
        """
        Generate comprehensive bias analysis report
        
        Args:
            df: DataFrame with transactions (must have 'amount', 'category', 'predicted' columns)
            save_path: Optional path to save report
            
        Returns:
            Report string
        """
        report = []
        report.append("="*70)
        report.append("⚖️  BIAS DETECTION AND FAIRNESS ANALYSIS REPORT")
        report.append("="*70)
        report.append("")
        
        # Amount-based bias
        report.append("1. AMOUNT-BASED BIAS ANALYSIS")
        report.append("-" * 70)
        amount_analysis = self.analyze_amount_bias(df)
        
        report.append(f"Bias Score (std dev of F1 across amount ranges): {amount_analysis['bias_score']:.4f}")
        report.append(f"Interpretation: {amount_analysis['interpretation']}")
        report.append("")
        
        for amount_range, metrics in amount_analysis['by_amount_range'].items():
            report.append(f"  {amount_range}:")
            report.append(f"    Samples: {metrics['count']}")
            report.append(f"    F1-Score: {metrics['f1_macro']:.3f}")
            report.append(f"    Accuracy: {metrics['accuracy']:.3f}")
            report.append("")
        
        # Category balance
        report.append("2. CATEGORY BALANCE ANALYSIS")
        report.append("-" * 70)
        category_analysis = self.analyze_category_balance(df['category'], df['predicted'])
        
        report.append(f"F1 Variance: {category_analysis['f1_variance']:.4f}")
        report.append(f"Min F1: {category_analysis['min_f1']:.3f}")
        report.append(f"Max F1: {category_analysis['max_f1']:.3f}")
        report.append(f"Interpretation: {category_analysis['interpretation']}")
        report.append("")
        
        # Merchant characteristics
        report.append("3. MERCHANT CHARACTERISTICS ANALYSIS")
        report.append("-" * 70)
        merchant_analysis = self.analyze_merchant_type_bias(df)
        
        for char_type, char_data in merchant_analysis.items():
            report.append(f"  {char_type}:")
            for value, metrics in char_data.items():
                report.append(f"    {value}: F1={metrics['f1_macro']:.3f} (n={metrics['count']})")
            report.append("")
        
        # Recommendations
        report.append("4. RECOMMENDATIONS")
        report.append("-" * 70)
        report.append(self._generate_recommendations(amount_analysis, category_analysis))
        report.append("")
        report.append("="*70)
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
        
        return report_str
    
    def _interpret_bias(self, bias_score: float) -> str:
        """Interpret bias score"""
        if bias_score < 0.05:
            return "✅ Low bias - Model performs consistently across amount ranges"
        elif bias_score < 0.10:
            return "⚠️ Moderate bias - Some variation in performance across amount ranges"
        else:
            return "❌ High bias - Significant performance variation across amount ranges"
    
    def _interpret_category_balance(self, variance: float) -> str:
        """Interpret category balance variance"""
        if variance < 0.01:
            return "✅ Well balanced - Consistent performance across categories"
        elif variance < 0.05:
            return "⚠️ Moderate imbalance - Some categories perform better than others"
        else:
            return "❌ Significant imbalance - Large performance gaps between categories"
    
    def _generate_recommendations(self, amount_analysis: Dict, category_analysis: Dict) -> str:
        """Generate bias mitigation recommendations"""
        recommendations = []
        
        if amount_analysis['bias_score'] > 0.05:
            recommendations.append("• Consider stratified sampling to balance amount ranges in training data")
            recommendations.append("• Incorporate amount-based features into the model")
        
        if category_analysis['f1_variance'] > 0.05:
            recommendations.append("• Collect more samples for underperforming categories")
            recommendations.append("• Apply class-weight balancing during training")
            recommendations.append("• Review and expand keyword lists for low-performing categories")
        
        if not recommendations:
            recommendations.append("✅ No major bias detected. Continue monitoring with new data.")
        
        return "\n".join(recommendations)


def demonstrate_bias_analysis(model, test_df: pd.DataFrame):
    """
    Demonstrate bias analysis on test data
    
    Args:
        model: Trained model
        test_df: Test DataFrame with 'description', 'amount', 'category' columns
    """
    # Generate predictions
    predictions, _ = model.predict(test_df['description'])
    test_df['predicted'] = predictions
    
    # Create analyzer and generate report
    analyzer = BiasAnalyzer(model)
    report = analyzer.generate_bias_report(test_df)
    
    print(report)
    
    return report
