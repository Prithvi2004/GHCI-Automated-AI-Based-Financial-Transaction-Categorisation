"""
Robustness Testing Module
Tests model resilience to noisy, corrupted, and adversarial inputs
"""

import pandas as pd
import numpy as np
import random
import string
from typing import List, Dict, Any
from sklearn.metrics import f1_score, accuracy_score


class RobustnessTest:
    """
    Test transaction categorization model robustness
    """
    
    def __init__(self, model):
        """
        Initialize robustness tester
        
        Args:
            model: Trained TransactionCategorizer instance
        """
        self.model = model
    
    def add_character_noise(self, text: str, noise_level: float = 0.1) -> str:
        """
        Add random character-level noise to text
        
        Args:
            text: Input text
            noise_level: Fraction of characters to corrupt (0.0 to 1.0)
            
        Returns:
            Noisy text
        """
        if not text or noise_level <= 0:
            return text
        
        chars = list(text)
        num_corruptions = int(len(chars) * noise_level)
        
        for _ in range(num_corruptions):
            if len(chars) == 0:
                break
            
            pos = random.randint(0, len(chars) - 1)
            noise_type = random.choice(['replace', 'delete', 'insert'])
            
            if noise_type == 'replace':
                chars[pos] = random.choice(string.ascii_uppercase + string.digits)
            elif noise_type == 'delete':
                chars.pop(pos)
            elif noise_type == 'insert':
                chars.insert(pos, random.choice(string.ascii_uppercase + string.digits))
        
        return ''.join(chars)
    
    def add_truncation(self, text: str, truncate_ratio: float = 0.3) -> str:
        """
        Truncate text by removing end portion
        
        Args:
            text: Input text
            truncate_ratio: Fraction of text to remove
            
        Returns:
            Truncated text
        """
        if not text or truncate_ratio <= 0:
            return text
        
        keep_len = max(1, int(len(text) * (1 - truncate_ratio)))
        return text[:keep_len]
    
    def add_case_variation(self, text: str) -> str:
        """
        Randomly vary case of text
        
        Args:
            text: Input text
            
        Returns:
            Case-varied text
        """
        if not text:
            return text
        
        case_type = random.choice(['lower', 'upper', 'title', 'random'])
        
        if case_type == 'lower':
            return text.lower()
        elif case_type == 'upper':
            return text.upper()
        elif case_type == 'title':
            return text.title()
        else:  # random
            return ''.join(
                c.upper() if random.random() > 0.5 else c.lower()
                for c in text
            )
    
    def add_extra_spaces(self, text: str) -> str:
        """
        Add random extra spaces
        
        Args:
            text: Input text
            
        Returns:
            Text with extra spaces
        """
        words = text.split()
        return '  '.join(words) if len(words) > 1 else text
    
    def add_special_characters(self, text: str) -> str:
        """
        Add random special characters
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters
        """
        specials = ['*', '#', '@', '!', '&', '%']
        num_specials = random.randint(1, 3)
        
        for _ in range(num_specials):
            pos = random.randint(0, len(text))
            text = text[:pos] + random.choice(specials) + text[pos:]
        
        return text
    
    def test_noise_robustness(self, test_df: pd.DataFrame, 
                            noise_levels: List[float] = None) -> Dict[str, Any]:
        """
        Test model performance under different noise levels
        
        Args:
            test_df: Test DataFrame with 'description' and 'category' columns
            noise_levels: List of noise levels to test
            
        Returns:
            Robustness test results
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
        
        results = {}
        
        # Baseline (no noise)
        baseline_preds, _ = self.model.predict(test_df['description'])
        baseline_f1 = f1_score(test_df['category'], baseline_preds, average='macro', zero_division=0)
        baseline_acc = accuracy_score(test_df['category'], baseline_preds)
        
        results[0.0] = {
            'f1_score': baseline_f1,
            'accuracy': baseline_acc,
            'degradation_f1': 0.0,
            'degradation_acc': 0.0
        }
        
        # Test with noise
        for noise_level in noise_levels:
            if noise_level == 0.0:
                continue
            
            # Apply noise
            noisy_texts = test_df['description'].apply(
                lambda x: self.add_character_noise(x, noise_level)
            )
            
            # Predict
            noisy_preds, _ = self.model.predict(noisy_texts)
            
            # Metrics
            noisy_f1 = f1_score(test_df['category'], noisy_preds, average='macro', zero_division=0)
            noisy_acc = accuracy_score(test_df['category'], noisy_preds)
            
            results[noise_level] = {
                'f1_score': noisy_f1,
                'accuracy': noisy_acc,
                'degradation_f1': baseline_f1 - noisy_f1,
                'degradation_acc': baseline_acc - noisy_acc
            }
        
        return results
    
    def test_input_variations(self, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Test model against various input perturbations
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Results for each perturbation type
        """
        variations = {
            'original': lambda x: x,
            'truncated': lambda x: self.add_truncation(x, 0.3),
            'lowercase': lambda x: x.lower(),
            'uppercase': lambda x: x.upper(),
            'extra_spaces': lambda x: self.add_extra_spaces(x),
            'special_chars': lambda x: self.add_special_characters(x),
            'character_noise': lambda x: self.add_character_noise(x, 0.1)
        }
        
        results = {}
        
        for var_name, var_func in variations.items():
            # Apply variation
            varied_texts = test_df['description'].apply(var_func)
            
            # Predict
            preds, confidences = self.model.predict(varied_texts)
            
            # Metrics
            f1 = f1_score(test_df['category'], preds, average='macro', zero_division=0)
            acc = accuracy_score(test_df['category'], preds)
            avg_conf = np.mean(confidences)
            
            results[var_name] = {
                'f1_score': f1,
                'accuracy': acc,
                'avg_confidence': avg_conf
            }
        
        return results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """
        Test model on edge cases and unusual inputs
        
        Returns:
            Edge case test results
        """
        edge_cases = [
            ("", "Empty string"),
            ("X", "Single character"),
            ("12345", "Only numbers"),
            ("@#$%^&*", "Only special chars"),
            ("A" * 200, "Very long text"),
            ("TRX PYMT ACH", "Abbreviations"),
            ("Ëxträordîñàry Çhârš", "Unicode characters"),
        ]
        
        results = []
        
        for text, description in edge_cases:
            try:
                pred, conf = self.model.predict([text])
                results.append({
                    'input': text[:50] + "..." if len(text) > 50 else text,
                    'description': description,
                    'predicted': pred[0],
                    'confidence': float(conf[0]),
                    'error': None
                })
            except Exception as e:
                results.append({
                    'input': text[:50] + "..." if len(text) > 50 else text,
                    'description': description,
                    'predicted': None,
                    'confidence': None,
                    'error': str(e)
                })
        
        return results
    
    def generate_robustness_report(self, test_df: pd.DataFrame, 
                                   save_path: str = None) -> str:
        """
        Generate comprehensive robustness testing report
        
        Args:
            test_df: Test DataFrame
            save_path: Optional path to save report
            
        Returns:
            Report string
        """
        report = []
        report.append("="*70)
        report.append("🛡️  ROBUSTNESS TESTING REPORT")
        report.append("="*70)
        report.append("")
        
        # Noise robustness
        report.append("1. NOISE ROBUSTNESS ANALYSIS")
        report.append("-" * 70)
        noise_results = self.test_noise_robustness(test_df)
        
        report.append(f"{'Noise Level':<15} {'F1-Score':<12} {'Accuracy':<12} {'F1 Degrad.':<15}")
        report.append("-" * 70)
        
        for noise_level, metrics in sorted(noise_results.items()):
            report.append(
                f"{noise_level:<15.0%} "
                f"{metrics['f1_score']:<12.3f} "
                f"{metrics['accuracy']:<12.3f} "
                f"{metrics['degradation_f1']:<15.3f}"
            )
        report.append("")
        
        # Input variations
        report.append("2. INPUT VARIATION ROBUSTNESS")
        report.append("-" * 70)
        variation_results = self.test_input_variations(test_df)
        
        report.append(f"{'Variation':<20} {'F1-Score':<12} {'Accuracy':<12} {'Avg Conf':<12}")
        report.append("-" * 70)
        
        for var_name, metrics in variation_results.items():
            report.append(
                f"{var_name:<20} "
                f"{metrics['f1_score']:<12.3f} "
                f"{metrics['accuracy']:<12.3f} "
                f"{metrics['avg_confidence']:<12.3f}"
            )
        report.append("")
        
        # Edge cases
        report.append("3. EDGE CASE HANDLING")
        report.append("-" * 70)
        edge_results = self.test_edge_cases()
        
        for result in edge_results:
            status = "✅" if result['error'] is None else "❌"
            report.append(f"{status} {result['description']:<25} ", )
            if result['error']:
                report.append(f"   Error: {result['error']}")
            else:
                report.append(f"   → {result['predicted']} (conf: {result['confidence']:.2f})")
        report.append("")
        
        # Summary
        report.append("4. ROBUSTNESS SUMMARY")
        report.append("-" * 70)
        report.append(self._generate_robustness_summary(noise_results, variation_results))
        report.append("")
        report.append("="*70)
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
        
        return report_str
    
    def _generate_robustness_summary(self, noise_results: Dict, 
                                    variation_results: Dict) -> str:
        """Generate robustness summary"""
        summary = []
        
        # Noise tolerance
        max_noise = max(noise_results.keys())
        degradation_at_max = noise_results[max_noise]['degradation_f1']
        
        if degradation_at_max < 0.05:
            summary.append("✅ Excellent noise tolerance - minimal performance degradation")
        elif degradation_at_max < 0.15:
            summary.append("✅ Good noise tolerance - acceptable performance under noise")
        else:
            summary.append("⚠️ Moderate noise tolerance - consider robustness improvements")
        
        # Variation tolerance
        original_f1 = variation_results.get('original', {}).get('f1_score', 0)
        min_f1 = min([v['f1_score'] for v in variation_results.values()])
        max_degradation = original_f1 - min_f1
        
        if max_degradation < 0.05:
            summary.append("✅ Highly robust to input variations")
        elif max_degradation < 0.15:
            summary.append("✅ Generally robust to input variations")
        else:
            summary.append("⚠️ Some sensitivity to input variations detected")
        
        return "\n".join(summary)


def demonstrate_robustness_testing(model, test_df: pd.DataFrame):
    """
    Demonstrate robustness testing
    
    Args:
        model: Trained model
        test_df: Test DataFrame
    """
    tester = RobustnessTest(model)
    report = tester.generate_robustness_report(test_df)
    
    print(report)
    
    return report
