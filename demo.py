"""
Interactive Command-Line Demo
Demonstrates transaction categorization with explainability and feedback
"""

import joblib
import pandas as pd
from src.preprocess import clean_text
from src.feedback import FeedbackSystem
import os


def main():
    print("\n" + "="*70)
    print("💰 TRANSACTION CATEGORIZATION SYSTEM - INTERACTIVE DEMO")
    print("="*70)
    print()
    
    # Load model
    if not os.path.exists("model.pkl"):
        print("❌ Model not found. Please run 'python train_model.py' first.")
        return
    
    categorizer = joblib.load("model.pkl")
    categories = categorizer.get_categories()
    feedback_system = FeedbackSystem()
    
    print("✅ Model loaded successfully")
    print(f"📋 Categories ({len(categories)}): {', '.join(categories)}")
    print()
    print("Instructions:")
    print("  • Enter a transaction description to categorize")
    print("  • Type 'stats' to see feedback statistics")
    print("  • Type 'examples' to see sample transactions")
    print("  • Type 'config' to view category configuration")
    print("  • Type 'quit' to exit")
    print()
    print("-" * 70)
    print()
    
    # Sample transactions for demonstration
    sample_transactions = [
        "AMAZON.COM*AB123CD456",
        "STARBUCKS STORE #12345",
        "SHELL OIL 87654321",
        "UBER * PENDING.UBER.COM",
        "WHOLE FOODS MKT #123",
        "CVS/PHARMACY #9876",
        "NETFLIX.COM",
        "DOORDASH*RESTAURANT",
        "ATM WITHDRAWAL 001234",
        "ZELLE TO JOHN SMITH"
    ]
    
    while True:
        raw_input = input("💳 Enter transaction: ").strip()
        
        if not raw_input:
            continue
        
        if raw_input.lower() == 'quit':
            print("\n👋 Thank you for using the Transaction Categorization System!")
            break
        
        if raw_input.lower() == 'stats':
            print("\n" + "="*70)
            stats = feedback_system.get_feedback_stats()
            print("📊 FEEDBACK STATISTICS")
            print("-" * 70)
            print(f"Total Feedback: {stats['total_feedback']}")
            print(f"Corrections: {stats['corrections']}")
            print(f"Accuracy: {stats['accuracy_on_feedback']:.2%}")
            print("="*70 + "\n")
            continue
        
        if raw_input.lower() == 'examples':
            print("\n" + "="*70)
            print("📝 SAMPLE TRANSACTIONS")
            print("-" * 70)
            for i, sample in enumerate(sample_transactions, 1):
                result = categorizer.predict_single(sample)
                print(f"{i:2d}. {sample:40s} → {result['category']:20s} ({result['confidence']:.0%})")
            print("="*70 + "\n")
            continue
        
        if raw_input.lower() == 'config':
            print("\n" + "="*70)
            print("⚙️  CATEGORY CONFIGURATION")
            print("-" * 70)
            for cat in categories:
                info = categorizer.get_category_info(cat)
                print(f"\n{cat}:")
                print(f"  Description: {info.get('description', 'N/A')}")
                print(f"  Keywords: {', '.join(info.get('keywords', [])[:5])}...")
            print("="*70 + "\n")
            continue
        
        # Process transaction
        cleaned = clean_text(raw_input)
        result = categorizer.predict_single(cleaned)
        
        # Display results
        print()
        print("─" * 70)
        print(f"📝 Transaction: {raw_input}")
        print(f"🎯 Category: {result['category']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"💡 Explanation: {result['explanation']}")
        print()
        print("🏆 Top 3 Predictions:")
        for i, (cat, prob) in enumerate(result['top_3_predictions'], 1):
            bar = "█" * int(prob * 30)
            print(f"   {i}. {cat:20s} {prob:6.2%} {bar}")
        print("─" * 70)
        
        # Feedback loop for low confidence
        if result['confidence'] < 0.85:
            print()
            print("⚠️  Low confidence detected!")
            print(f"📋 Available categories: {', '.join(categories)}")
            print()
            correct = input("✏️  Enter correct category (or press Enter to skip): ").strip()
            
            if correct and correct in categories:
                feedback_system.add_feedback(
                    description=cleaned,
                    predicted_category=result['category'],
                    true_category=correct,
                    confidence=result['confidence']
                )
                print("✅ Feedback recorded. Thank you!")
                print("   Run 'stats' to see updated feedback statistics")
            elif correct and correct not in categories:
                print(f"❌ '{correct}' is not a valid category")
        
        print()
    
    # Show final feedback stats
    stats = feedback_system.get_feedback_stats()
    if stats['total_feedback'] > 0:
        print("\n" + "="*70)
        print("📊 SESSION SUMMARY")
        print("-" * 70)
        print(f"Feedback collected: {stats['total_feedback']}")
        print(f"Corrections made: {stats['corrections']}")
        
        if stats['total_feedback'] >= 50:
            print("\n💡 Tip: You have enough feedback to retrain the model!")
            print("   Consider running model retraining to improve accuracy.")
        
        print("="*70)
        print()
        
        # Export feedback
        feedback_system.export_feedback_for_review("feedback_review.csv")
        print("� Feedback exported to: feedback_review.csv")
    
    print()


if __name__ == "__main__":
    main()
