"""
Human-in-the-Loop Feedback System
Collects user feedback and enables active learning
"""

import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any
import joblib


class FeedbackSystem:
    """
    Manages user feedback collection and model retraining
    """
    
    def __init__(self, feedback_file: str = "data/feedback.csv"):
        """
        Initialize feedback system
        
        Args:
            feedback_file: Path to feedback storage file
        """
        self.feedback_file = feedback_file
        self._ensure_feedback_file()
    
    def _ensure_feedback_file(self):
        """Ensure feedback file exists with proper structure"""
        if not os.path.exists(self.feedback_file):
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
            df = pd.DataFrame(columns=[
                'timestamp', 'description', 'amount', 'predicted_category',
                'true_category', 'confidence', 'user_id'
            ])
            df.to_csv(self.feedback_file, index=False)
    
    def add_feedback(self, description: str, predicted_category: str,
                    true_category: str, confidence: float = None,
                    amount: float = None, user_id: str = "anonymous"):
        """
        Add user feedback for a transaction
        
        Args:
            description: Transaction description
            predicted_category: Model's prediction
            true_category: User-corrected category
            confidence: Prediction confidence
            amount: Transaction amount
            user_id: User identifier
        """
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'amount': amount,
            'predicted_category': predicted_category,
            'true_category': true_category,
            'confidence': confidence,
            'user_id': user_id
        }
        
        # Append to file
        df = pd.read_csv(self.feedback_file)
        df = pd.concat([df, pd.DataFrame([feedback_entry])], ignore_index=True)
        df.to_csv(self.feedback_file, index=False)
        
        return True
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected feedback
        
        Returns:
            Feedback statistics dictionary
        """
        df = pd.read_csv(self.feedback_file)
        
        if len(df) == 0:
            return {
                'total_feedback': 0,
                'corrections': 0,
                'accuracy_on_feedback': 0.0
            }
        
        # Calculate stats
        corrections = df[df['predicted_category'] != df['true_category']]
        
        stats = {
            'total_feedback': len(df),
            'corrections': len(corrections),
            'accuracy_on_feedback': 1.0 - (len(corrections) / len(df)) if len(df) > 0 else 0.0,
            'most_confused_pairs': self._get_confused_pairs(df),
            'low_confidence_corrections': len(corrections[corrections['confidence'] < 0.8]) if 'confidence' in corrections else 0,
            'feedback_by_category': df['true_category'].value_counts().to_dict()
        }
        
        return stats
    
    def _get_confused_pairs(self, df: pd.DataFrame, top_n: int = 5) -> List[tuple]:
        """Get most commonly confused category pairs"""
        corrections = df[df['predicted_category'] != df['true_category']]
        
        if len(corrections) == 0:
            return []
        
        pairs = corrections.groupby(['predicted_category', 'true_category']).size()
        pairs = pairs.sort_values(ascending=False).head(top_n)
        
        return [(pred, true, count) for (pred, true), count in pairs.items()]
    
    def get_feedback_data(self) -> pd.DataFrame:
        """
        Get all feedback data
        
        Returns:
            DataFrame with all feedback
        """
        return pd.read_csv(self.feedback_file)
    
    def get_correction_candidates(self, min_confidence_threshold: float = 0.85) -> pd.DataFrame:
        """
        Get transactions that likely need correction (low confidence)
        
        Args:
            min_confidence_threshold: Confidence threshold
            
        Returns:
            DataFrame with low-confidence predictions
        """
        df = pd.read_csv(self.feedback_file)
        
        if 'confidence' not in df.columns or len(df) == 0:
            return pd.DataFrame()
        
        return df[df['confidence'] < min_confidence_threshold].sort_values('confidence')
    
    def retrain_with_feedback(self, model, original_train_data: pd.DataFrame,
                             min_feedback_samples: int = 50) -> tuple:
        """
        Retrain model incorporating user feedback
        
        Args:
            model: Current model instance
            original_train_data: Original training DataFrame
            min_feedback_samples: Minimum feedback samples before retraining
            
        Returns:
            (retrained_model, training_stats)
        """
        from src.preprocess import preprocess_dataframe
        
        feedback_df = self.get_feedback_data()
        
        if len(feedback_df) < min_feedback_samples:
            return model, {
                'retrained': False,
                'reason': f'Insufficient feedback samples ({len(feedback_df)} < {min_feedback_samples})'
            }
        
        # Prepare feedback data
        feedback_df = feedback_df.rename(columns={'true_category': 'category'})
        feedback_df = preprocess_dataframe(feedback_df, text_col='description')
        
        # Combine with original training data
        combined_train = pd.concat([
            original_train_data[['description', 'category']],
            feedback_df[['description', 'category']]
        ], ignore_index=True)
        
        # Remove duplicates
        combined_train = combined_train.drop_duplicates(subset=['description'])
        
        # Retrain model
        print(f"🔄 Retraining with {len(feedback_df)} feedback samples...")
        model.fit(combined_train['description'], combined_train['category'])
        
        stats = {
            'retrained': True,
            'original_samples': len(original_train_data),
            'feedback_samples': len(feedback_df),
            'total_samples': len(combined_train),
            'timestamp': datetime.now().isoformat()
        }
        
        return model, stats
    
    def export_feedback_for_review(self, output_file: str = "feedback_review.csv"):
        """
        Export feedback in human-readable format for review
        
        Args:
            output_file: Output CSV file path
        """
        df = self.get_feedback_data()
        
        if len(df) == 0:
            print("⚠️ No feedback data to export")
            return
        
        # Add helpful columns
        df['needs_review'] = df['predicted_category'] != df['true_category']
        df['confidence_level'] = pd.cut(
            df['confidence'],
            bins=[0, 0.7, 0.85, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Reorder columns for readability
        cols = ['timestamp', 'description', 'predicted_category', 'true_category',
                'needs_review', 'confidence', 'confidence_level', 'amount', 'user_id']
        df = df[[c for c in cols if c in df.columns]]
        
        df.to_csv(output_file, index=False)
        print(f"✅ Feedback exported to {output_file}")
    
    def generate_feedback_report(self, save_path: str = None) -> str:
        """
        Generate comprehensive feedback analysis report
        
        Args:
            save_path: Optional path to save report
            
        Returns:
            Report string
        """
        stats = self.get_feedback_stats()
        
        report = []
        report.append("="*70)
        report.append("💬 FEEDBACK ANALYSIS REPORT")
        report.append("="*70)
        report.append("")
        
        report.append("1. OVERALL STATISTICS")
        report.append("-" * 70)
        report.append(f"  Total Feedback Collected:     {stats['total_feedback']}")
        report.append(f"  Corrections Made:             {stats['corrections']}")
        report.append(f"  Accuracy on Feedback:         {stats['accuracy_on_feedback']:.2%}")
        report.append(f"  Low-Conf Corrections:         {stats['low_confidence_corrections']}")
        report.append("")
        
        if stats['most_confused_pairs']:
            report.append("2. MOST CONFUSED CATEGORY PAIRS")
            report.append("-" * 70)
            for pred, true, count in stats['most_confused_pairs']:
                report.append(f"  {pred} → {true}: {count} times")
            report.append("")
        
        report.append("3. FEEDBACK BY CATEGORY")
        report.append("-" * 70)
        for category, count in sorted(stats['feedback_by_category'].items(), 
                                     key=lambda x: x[1], reverse=True):
            report.append(f"  {category}: {count}")
        report.append("")
        
        report.append("4. RECOMMENDATIONS")
        report.append("-" * 70)
        
        if stats['total_feedback'] < 50:
            report.append("  • Continue collecting feedback to enable model retraining")
        else:
            report.append("  • ✅ Sufficient feedback for retraining - consider running retrain")
        
        if stats['corrections'] > stats['total_feedback'] * 0.2:
            report.append("  • ⚠️ High correction rate - review model performance")
        
        report.append("")
        report.append("="*70)
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
        
        return report_str


def demonstrate_feedback_system():
    """Demonstrate feedback system functionality"""
    print("\n" + "="*60)
    print("💬 FEEDBACK SYSTEM DEMONSTRATION")
    print("="*60)
    
    feedback_sys = FeedbackSystem()
    
    # Simulate some feedback
    samples = [
        ("AMAZON.COM*1234", "Entertainment", "Shopping", 0.65, 45.99),
        ("STARBUCKS #456", "Coffee/Dining", "Coffee/Dining", 0.95, 8.50),
        ("UNKNOWN MERCHANT", "Shopping", "Groceries", 0.55, 67.80),
    ]
    
    print("\n📝 Adding sample feedback...")
    for desc, pred, true, conf, amt in samples:
        feedback_sys.add_feedback(desc, pred, true, conf, amt)
        status = "✅" if pred == true else "❌"
        print(f"  {status} {desc}: {pred} → {true} (conf: {conf:.2f})")
    
    # Show stats
    print("\n📊 Feedback Statistics:")
    stats = feedback_sys.get_feedback_stats()
    print(f"  Total feedback: {stats['total_feedback']}")
    print(f"  Corrections: {stats['corrections']}")
    print(f"  Accuracy: {stats['accuracy_on_feedback']:.2%}")
    
    # Generate report
    report = feedback_sys.generate_feedback_report()
    print("\n" + report)
