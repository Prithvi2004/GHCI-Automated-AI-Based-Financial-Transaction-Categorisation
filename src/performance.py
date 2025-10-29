"""
Performance Benchmarking Module
Measures throughput, latency, and batch processing performance
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import psutil
import os


class PerformanceBenchmark:
    """
    Benchmark transaction categorization performance
    """
    
    def __init__(self, model):
        """
        Initialize benchmark with model
        
        Args:
            model: Trained TransactionCategorizer instance
        """
        self.model = model
    
    def measure_latency(self, texts: List[str], num_iterations: int = 100) -> Dict[str, float]:
        """
        Measure single-transaction prediction latency
        
        Args:
            texts: Sample texts for testing
            num_iterations: Number of iterations for averaging
            
        Returns:
            Latency statistics in milliseconds
        """
        latencies = []
        
        for _ in range(num_iterations):
            # Pick random text
            text = texts[np.random.randint(0, len(texts))]
            
            start = time.perf_counter()
            self.model.predict([text])
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'std_ms': np.std(latencies)
        }
    
    def measure_throughput(self, texts: List[str], batch_sizes: List[int] = None) -> Dict[int, Dict[str, float]]:
        """
        Measure batch processing throughput
        
        Args:
            texts: Sample texts for testing
            batch_sizes: List of batch sizes to test
            
        Returns:
            Throughput metrics for each batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 10, 50, 100, 500, 1000]
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(texts):
                continue
            
            # Prepare batch
            batch = texts[:batch_size]
            
            # Warm-up
            self.model.predict(batch)
            
            # Measure
            num_runs = max(1, 100 // batch_size)  # Adjust runs based on batch size
            times = []
            
            for _ in range(num_runs):
                start = time.perf_counter()
                self.model.predict(batch)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time  # transactions per second
            
            results[batch_size] = {
                'avg_time_sec': avg_time,
                'throughput_tps': throughput,
                'time_per_transaction_ms': (avg_time / batch_size) * 1000
            }
        
        return results
    
    def measure_memory_usage(self, texts: List[str], batch_size: int = 1000) -> Dict[str, float]:
        """
        Measure memory usage during prediction
        
        Args:
            texts: Sample texts
            batch_size: Batch size for testing
            
        Returns:
            Memory usage statistics in MB
        """
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Get baseline memory
        process = psutil.Process(os.getpid())
        baseline_mb = process.memory_info().rss / 1024 / 1024
        
        # Perform prediction
        batch = texts[:min(batch_size, len(texts))]
        self.model.predict(batch)
        
        # Measure peak memory
        peak_mb = process.memory_info().rss / 1024 / 1024
        
        # Memory used
        used_mb = peak_mb - baseline_mb
        
        return {
            'baseline_mb': baseline_mb,
            'peak_mb': peak_mb,
            'used_mb': used_mb,
            'per_transaction_kb': (used_mb * 1024) / len(batch)
        }
    
    def run_full_benchmark(self, test_texts: List[str], save_path: str = None) -> str:
        """
        Run comprehensive performance benchmark
        
        Args:
            test_texts: Sample texts for benchmarking
            save_path: Optional path to save results
            
        Returns:
            Benchmark report string
        """
        report = []
        report.append("="*70)
        report.append("⚡ PERFORMANCE BENCHMARK REPORT")
        report.append("="*70)
        report.append("")
        
        # Latency
        report.append("1. LATENCY ANALYSIS (Single Transaction)")
        report.append("-" * 70)
        latency = self.measure_latency(test_texts, num_iterations=100)
        report.append(f"  Mean Latency:       {latency['mean_ms']:.2f} ms")
        report.append(f"  Median Latency:     {latency['median_ms']:.2f} ms")
        report.append(f"  P95 Latency:        {latency['p95_ms']:.2f} ms")
        report.append(f"  P99 Latency:        {latency['p99_ms']:.2f} ms")
        report.append(f"  Std Dev:            {latency['std_ms']:.2f} ms")
        report.append("")
        
        # Throughput
        report.append("2. THROUGHPUT ANALYSIS (Batch Processing)")
        report.append("-" * 70)
        batch_sizes = [1, 10, 50, 100, 500]
        throughput = self.measure_throughput(test_texts, batch_sizes)
        
        report.append(f"{'Batch Size':<15} {'Avg Time (s)':<15} {'Throughput (TPS)':<20} {'Time/Txn (ms)':<15}")
        report.append("-" * 70)
        
        for batch_size, metrics in throughput.items():
            report.append(
                f"{batch_size:<15} "
                f"{metrics['avg_time_sec']:<15.3f} "
                f"{metrics['throughput_tps']:<20.1f} "
                f"{metrics['time_per_transaction_ms']:<15.2f}"
            )
        report.append("")
        
        # Memory usage
        report.append("3. MEMORY USAGE ANALYSIS")
        report.append("-" * 70)
        memory = self.measure_memory_usage(test_texts, batch_size=1000)
        report.append(f"  Baseline Memory:    {memory['baseline_mb']:.2f} MB")
        report.append(f"  Peak Memory:        {memory['peak_mb']:.2f} MB")
        report.append(f"  Memory Used:        {memory['used_mb']:.2f} MB")
        report.append(f"  Per Transaction:    {memory['per_transaction_kb']:.2f} KB")
        report.append("")
        
        # Summary
        report.append("4. SUMMARY & RECOMMENDATIONS")
        report.append("-" * 70)
        report.append(self._generate_performance_summary(latency, throughput))
        report.append("")
        report.append("="*70)
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
        
        return report_str
    
    def _generate_performance_summary(self, latency: Dict, throughput: Dict) -> str:
        """Generate performance summary and recommendations"""
        summary = []
        
        mean_latency = latency['mean_ms']
        
        # Latency assessment
        if mean_latency < 10:
            summary.append("✅ Excellent latency - suitable for real-time applications")
        elif mean_latency < 50:
            summary.append("✅ Good latency - suitable for most interactive applications")
        elif mean_latency < 200:
            summary.append("⚠️ Moderate latency - acceptable for batch processing")
        else:
            summary.append("❌ High latency - consider model optimization")
        
        # Throughput assessment
        max_throughput = max([m['throughput_tps'] for m in throughput.values()])
        
        if max_throughput > 1000:
            summary.append("✅ High throughput - can handle large-scale production loads")
        elif max_throughput > 100:
            summary.append("✅ Good throughput - suitable for medium-scale applications")
        else:
            summary.append("⚠️ Consider batch processing to improve throughput")
        
        # Recommendations
        summary.append("\nRecommendations:")
        summary.append("• Use batch processing (100-500 transactions) for optimal throughput")
        summary.append("• Consider caching for frequently seen transaction patterns")
        summary.append("• Monitor memory usage in production for sustained loads")
        
        return "\n".join(summary)
    
    def save_benchmark_results(self, test_texts: List[str], output_file: str = "benchmark_results.json"):
        """
        Save detailed benchmark results to JSON
        
        Args:
            test_texts: Sample texts for benchmarking
            output_file: Output file path
        """
        import json
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_test_samples': len(test_texts),
            'latency': self.measure_latency(test_texts, num_iterations=100),
            'throughput': {
                str(k): v for k, v in self.measure_throughput(test_texts).items()
            },
            'memory': self.measure_memory_usage(test_texts)
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Benchmark results saved to {output_file}")


def demonstrate_benchmarking(model, test_texts: List[str]):
    """
    Demonstrate performance benchmarking
    
    Args:
        model: Trained model
        test_texts: Test transaction descriptions
    """
    benchmark = PerformanceBenchmark(model)
    report = benchmark.run_full_benchmark(test_texts)
    
    print(report)
    
    # Save detailed results
    benchmark.save_benchmark_results(test_texts)
    
    return report
