"""
End-to-End Pipeline Runner
Executes complete training, evaluation, and demo workflow
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors"""
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} - Complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  AI-Based Financial Transaction Categorization                  ║
    ║  End-to-End Pipeline Runner                                      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nThis script will run the complete pipeline:")
    print("  1. Generate synthetic training data")
    print("  2. Train ensemble model")
    print("  3. Run comprehensive evaluation")
    print("  4. Generate all reports")
    print()
    
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Step 1: Generate data
    if not os.path.exists("data/train_transactions.csv"):
        if not run_command("python -m src.data_generator", 
                          "Step 1/4: Generating Synthetic Data"):
            return
    else:
        print("\n✅ Step 1/4: Training data already exists")
    
    # Step 2: Train model
    if not run_command("python train_model.py", 
                      "Step 2/4: Training Model"):
        return
    
    # Step 3: Evaluate model
    if not run_command("python evaluate_model.py", 
                      "Step 3/4: Running Comprehensive Evaluation"):
        return
    
    # Step 4: Summary
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    print()
    print("✅ Generated Files:")
    print("   • model.pkl - Trained ensemble model")
    print("   • evaluation_report.json - Detailed metrics")
    print("   • bias_report.txt - Fairness analysis")
    print("   • performance_report.txt - Benchmarking results")
    print("   • robustness_report.txt - Noise tolerance testing")
    print("   • benchmark_results.json - Performance data")
    print()
    print("📂 Data Files:")
    print("   • data/train_transactions.csv - Training dataset")
    print("   • data/test_transactions.csv - Test dataset")
    print()
    print("🚀 Next Steps:")
    print("   1. Run 'python demo.py' for interactive CLI demo")
    print("   2. Run 'streamlit run app.py' for web interface")
    print("   3. Review generated reports for detailed analysis")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
