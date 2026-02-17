#!/usr/bin/env python
"""
Getting Started Script for Data Quality Checks
Demonstrates basic usage of the library with a simple example.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_quality_checker import DataQualityChecker


def main():
    print("=" * 70)
    print("Welcome to Data Quality Checks for ML/NLP!")
    print("=" * 70)
    
    try:
        # Initialize the quality checker
        print("\n1. Initializing DataQualityChecker...")
        checker = DataQualityChecker()
        print("   ✓ Initialized successfully!")
        
        # Simple example
        print("\n2. Running a simple evaluation example...")
        
        prediction = "The machine learning model achieved excellent performance"
        reference = "ML model performed very well"
        
        print(f"\n   Prediction: {prediction}")
        print(f"   Reference:  {reference}")
        
        # Compute ROUGE
        print("\n3. Computing ROUGE metrics...")
        rouge_scores = checker.compute_rouge(prediction, reference)
        
        print("   ROUGE Scores:")
        for metric, score in rouge_scores.items():
            print(f"     {metric}: {score:.4f}")
        
        # Compute BLEU
        print("\n4. Computing BLEU score...")
        bleu_scores = checker.compute_bleu([prediction], [[reference]])
        print(f"   BLEU Score: {bleu_scores.get('bleu', 0):.4f}")
        
        # All metrics
        print("\n5. Computing all metrics combined...")
        all_metrics = checker.compute_all_metrics(prediction, reference)
        print(checker.format_results(all_metrics))
        
        print("\n" + "=" * 70)
        print("✓ Getting started example completed successfully!")
        print("=" * 70)
        
        print("\nNext steps:")
        print("  1. Check QUICKSTART.md for common usage patterns")
        print("  2. Run examples: python examples/example_1_basic_usage.py")
        print("  3. Read README.md for comprehensive documentation")
        print("  4. Explore ADVANCED_USAGE.md for advanced scenarios")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Verify dependencies: pip install -r requirements.txt")
        print("  2. Check that evaluate is installed: pip install evaluate")
        print("  3. Review error message above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
