"""
Advanced Model Training Pipeline
Trains ensemble model with comprehensive evaluation
"""

import pandas as pd
import joblib
import os
from src.preprocess import preprocess_dataframe
from src.model import TransactionCategorizer
from src.data_generator import TransactionDataGenerator


def main():
    print("="*70)
    print("🚀 TRANSACTION CATEGORIZATION - MODEL TRAINING")
    print("="*70)
    print()
    
    # Step 1: Generate synthetic data if needed
    if not os.path.exists("data/train_transactions.csv"):
        print("📊 Generating synthetic training data...")
        generator = TransactionDataGenerator()
        generator.save_datasets()
        print()
    
    # Step 2: Load training data
    print("📂 Loading training data...")
    train_df = pd.read_csv("data/train_transactions.csv")
    print(f"   Loaded {len(train_df)} training samples")
    print(f"   Categories: {train_df['category'].nunique()}")
    print(f"   Category distribution:")
    for cat, count in train_df['category'].value_counts().items():
        print(f"      {cat}: {count}")
    print()
    
    # Step 3: Preprocess
    print("🔧 Preprocessing data...")
    train_df = preprocess_dataframe(train_df, text_col='description')
    print("   ✅ Preprocessing complete")
    print()
    
    # Step 4: Train model
    print("🤖 Training advanced ensemble model...")
    print("   - Random Forest (200 estimators)")
    print("   - Gradient Boosting (150 estimators)")
    print("   - Logistic Regression (multinomial)")
    print("   - TF-IDF features (word + character n-grams)")
    print()
    
    categorizer = TransactionCategorizer()
    categorizer.fit(train_df['description'], train_df['category'])
    
    print("   ✅ Model training complete")
    print()
    
    # Step 5: Quick validation on training set
    print("📈 Quick training set validation...")
    train_preds, train_confs = categorizer.predict(train_df['description'])
    train_accuracy = (train_preds == train_df['category']).mean()
    print(f"   Training accuracy: {train_accuracy:.3f}")
    print(f"   Average confidence: {train_confs.mean():.3f}")
    print()
    
    # Step 6: Save model
    model_path = "model.pkl"
    joblib.dump(categorizer, model_path)
    print(f"💾 Model saved to: {model_path}")
    print()
    
    print("="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Run 'python evaluate_model.py' for comprehensive evaluation")
    print("  2. Run 'python demo.py' for interactive demo")
    print("  3. Run 'python app.py' for web interface (if Streamlit installed)")
    print()


if __name__ == "__main__":
    main()
