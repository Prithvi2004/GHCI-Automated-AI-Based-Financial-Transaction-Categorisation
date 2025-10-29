"""
Comprehensive Model Evaluation Pipeline
Generates detailed metrics, confusion matrix, and analysis reports
"""

import pandas as pd
import joblib
import json
import os
from src.preprocess import preprocess_dataframe
from src.evaluate import generate_evaluation_report
from src.explainability import TransactionExplainer, demonstrate_explainability
from src.bias_detection import BiasAnalyzer, demonstrate_bias_analysis
from src.performance import PerformanceBenchmark, demonstrate_benchmarking
from src.robustness import RobustnessTest, demonstrate_robustness_testing


def main():
    print("="*70)
    print("📊 TRANSACTION CATEGORIZATION - COMPREHENSIVE EVALUATION")
    print("="*70)
    print()
    
    # Load test data and model
    print("📂 Loading test data and model...")
    
    if not os.path.exists("data/test_transactions.csv"):
        print("❌ Test data not found. Please run train_model.py first.")
        return
    
    if not os.path.exists("model.pkl"):
        print("❌ Model not found. Please run train_model.py first.")
        return
    
    test_df = pd.read_csv("data/test_transactions.csv")
    test_df = preprocess_dataframe(test_df, text_col='description')
    categorizer = joblib.load("model.pkl")
    
    print(f"   ✅ Loaded {len(test_df)} test samples")
    print(f"   ✅ Model loaded successfully")
    print()
    
    # Generate predictions
    print("🔮 Generating predictions...")
    predictions, confidences = categorizer.predict(test_df['description'])
    test_df['predicted'] = predictions
    test_df['confidence'] = confidences
    print("   ✅ Predictions complete")
    print()
    
    # ========== Core Evaluation ==========
    print("="*70)
    print("1️⃣  CORE PERFORMANCE METRICS")
    print("="*70)
    
    report = generate_evaluation_report(
        test_df['category'], 
        predictions, 
        categorizer.get_categories()
    )
    
    print(f"\n📈 MACRO F1-SCORE: {report['macro_f1']:.4f}")
    print(f"   Target: ≥ 0.90 | Status: {'✅ ACHIEVED' if report['macro_f1'] >= 0.90 else '⚠️ BELOW TARGET'}")
    print()
    
    print("📊 Per-Category F1 Scores:")
    for cat, f1 in sorted(report['per_class_f1'].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(f1 * 20)
        print(f"   {cat:20s} {f1:.3f} {bar}")
    print()
    
    # Save evaluation report
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("💾 Detailed metrics saved to: evaluation_report.json")
    print()
    
    # ========== Explainability Analysis ==========
    print("="*70)
    print("2️⃣  EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    sample_transactions = test_df['description'].sample(min(5, len(test_df))).tolist()
    demonstrate_explainability(categorizer, sample_transactions)
    print()
    
    # ========== Bias Detection ==========
    print("="*70)
    print("3️⃣  BIAS DETECTION & FAIRNESS ANALYSIS")
    print("="*70)
    
    bias_report = demonstrate_bias_analysis(categorizer, test_df)
    
    with open("bias_report.txt", "w") as f:
        f.write(bias_report)
    print("💾 Bias analysis saved to: bias_report.txt")
    print()
    
    # ========== Performance Benchmarking ==========
    print("="*70)
    print("4️⃣  PERFORMANCE BENCHMARKING")
    print("="*70)
    
    perf_report = demonstrate_benchmarking(categorizer, test_df['description'].tolist())
    
    with open("performance_report.txt", "w") as f:
        f.write(perf_report)
    print("💾 Performance benchmark saved to: performance_report.txt")
    print()
    
    # ========== Robustness Testing ==========
    print("="*70)
    print("5️⃣  ROBUSTNESS TESTING")
    print("="*70)
    
    robustness_report = demonstrate_robustness_testing(categorizer, test_df)
    
    with open("robustness_report.txt", "w") as f:
        f.write(robustness_report)
    print("💾 Robustness analysis saved to: robustness_report.txt")
    print()
    
    # ========== Summary ==========
    print("="*70)
    print("📋 EVALUATION SUMMARY")
    print("="*70)
    print()
    print("✅ Core Metrics:")
    print(f"   • Macro F1-Score: {report['macro_f1']:.4f}")
    print(f"   • Accuracy: {report['classification_report']['accuracy']:.4f}")
    print(f"   • Weighted F1: {report['classification_report']['weighted avg']['f1-score']:.4f}")
    print()
    print("✅ Generated Reports:")
    print("   • evaluation_report.json - Detailed metrics and confusion matrix")
    print("   • bias_report.txt - Fairness analysis")
    print("   • performance_report.txt - Throughput and latency benchmarks")
    print("   • robustness_report.txt - Noise and variation testing")
    print("   • benchmark_results.json - Detailed performance data")
    print()
    print("✅ Key Findings:")
    
    if report['macro_f1'] >= 0.90:
        print("   🎯 Macro F1-score exceeds 0.90 target")
    else:
        print(f"   ⚠️ Macro F1-score ({report['macro_f1']:.3f}) below 0.90 target")
        print("      Consider: more training data, feature engineering, or hyperparameter tuning")
    
    low_conf_pct = (confidences < 0.85).sum() / len(confidences)
    print(f"   📊 Low confidence predictions: {low_conf_pct:.1%}")
    
    if low_conf_pct > 0.2:
        print("      ⚠️ Consider human-in-the-loop feedback for low-confidence predictions")
    
    print()
    print("="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
