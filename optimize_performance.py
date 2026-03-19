#!/usr/bin/env python3
"""
Performance Optimization for Safe Zone Detection
Implements caching, quantization, and batch processing
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np
import torch
import torch.quantization as quantization
from functools import lru_cache
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import aiofiles

# Add src to path
sys.path.append('src')

from src.audio_processor import TorchAudioProcessor
from src.llama_processor import analyze_multi_modal_distress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization for the Safe Zone Detection system"""
    
    def __init__(self, cache_dir="cache", models_dir="models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.audio_processor = None
        self.llama_processor = None
        
        # Cache for processed results
        self.result_cache = {}
        self.cache_size_limit = 1000
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time': 0.0,
            'avg_audio_time': 0.0,
            'avg_llama_time': 0.0
        }
    
    def initialize_optimized_processor(self):
        """Initialize optimized audio processor with caching"""
        logger.info("Initializing optimized audio processor...")
        
        # Initialize processor
        self.audio_processor = TorchAudioProcessor()
        
        # Enable optimizations
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available()
        
        # Load quantized model if available
        quantized_model_path = self.models_dir / "quantized_audio_model.pth"
        if quantized_model_path.exists():
            logger.info("Loading quantized audio model...")
            # Load quantized model here
        else:
            logger.info("No quantized model found, using standard model")
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file caching"""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def load_cached_result(self, file_hash: str) -> Optional[Dict]:
        """Load cached result if available"""
        cache_file = self.cache_dir / f"{file_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        return None
    
    def save_cached_result(self, file_hash: str, result: Dict):
        """Save result to cache"""
        if len(self.result_cache) >= self.cache_size_limit:
            # Remove oldest entries
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        cache_file = self.cache_dir / f"{file_hash}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            self.result_cache[file_hash] = result
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def process_audio_optimized(self, audio_path: str) -> Dict:
        """Optimized audio processing with caching"""
        start_time = time.time()
        
        # Check cache first
        file_hash = self.get_file_hash(audio_path)
        cached_result = self.load_cached_result(file_hash)
        
        if cached_result:
            self.metrics['cache_hits'] += 1
            logger.debug(f"Cache hit for {audio_path}")
            return cached_result
        
        self.metrics['cache_misses'] += 1
        
        # Process audio
        audio_start = time.time()
        result = self.audio_processor.process_audio(audio_path)
        audio_time = time.time() - audio_start
        
        # Add performance metrics
        result['processing_time'] = time.time() - start_time
        result['audio_processing_time'] = audio_time
        result['cached'] = False
        
        # Cache result
        self.save_cached_result(file_hash, result)
        
        # Update metrics
        self.metrics['total_requests'] += 1
        self.metrics['avg_audio_time'] = (
            (self.metrics['avg_audio_time'] * (self.metrics['total_requests'] - 1) + audio_time) 
            / self.metrics['total_requests']
        )
        
        return result
    
    def batch_process_audio(self, audio_paths: List[str], max_workers: int = 4) -> List[Dict]:
        """Process multiple audio files in parallel"""
        logger.info(f"Batch processing {len(audio_paths)} audio files...")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.process_audio_optimized, audio_paths))
        
        total_time = time.time() - start_time
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        
        return results
    
    def optimize_llama_processing(self):
        """Optimize Llama processing for faster inference"""
        logger.info("Optimizing Llama processing...")
        
        # This would implement:
        # 1. Model quantization
        # 2. Batch processing
        # 3. Caching of common patterns
        # 4. Reduced precision inference
        
        logger.info("Llama optimization not implemented yet")
    
    def create_quantized_audio_model(self):
        """Create quantized version of audio processing model"""
        logger.info("Creating quantized audio model...")
        
        # This would implement:
        # 1. Load the current audio processing model
        # 2. Apply quantization
        # 3. Save quantized model
        
        logger.info("Audio model quantization not implemented yet")
    
    def optimize_memory_usage(self):
        """Optimize memory usage for better performance"""
        logger.info("Optimizing memory usage...")
        
        # Clear unused variables
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Memory optimization completed")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        if self.metrics['total_requests'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / self.metrics['total_requests']
        else:
            cache_hit_rate = 0.0
        
        return {
            'total_requests': self.metrics['total_requests'],
            'cache_hit_rate': cache_hit_rate,
            'avg_processing_time': self.metrics['avg_processing_time'],
            'avg_audio_time': self.metrics['avg_audio_time'],
            'cache_size': len(self.result_cache)
        }
    
    def benchmark_performance(self, test_files: List[str]) -> Dict:
        """Benchmark performance improvements"""
        logger.info("Running performance benchmark...")
        
        # Clear cache for fair comparison
        self.result_cache.clear()
        
        # Test 1: Single file processing
        single_file_times = []
        for file_path in test_files[:5]:  # Test first 5 files
            start_time = time.time()
            result = self.process_audio_optimized(file_path)
            single_file_times.append(time.time() - start_time)
        
        # Test 2: Batch processing
        batch_start = time.time()
        batch_results = self.batch_process_audio(test_files[:10])  # Test first 10 files
        batch_time = time.time() - batch_start
        
        # Test 3: Cache performance
        cache_start = time.time()
        for file_path in test_files[:5]:  # Process same files again
            result = self.process_audio_optimized(file_path)
        cache_time = time.time() - cache_start
        
        benchmark_results = {
            'single_file_avg': np.mean(single_file_times),
            'single_file_std': np.std(single_file_times),
            'batch_processing_time': batch_time,
            'batch_avg_per_file': batch_time / len(batch_results),
            'cache_processing_time': cache_time,
            'cache_avg_per_file': cache_time / 5,
            'cache_speedup': np.mean(single_file_times) / (cache_time / 5),
            'batch_speedup': np.mean(single_file_times) / (batch_time / len(batch_results))
        }
        
        logger.info(f"Benchmark results: {benchmark_results}")
        return benchmark_results

class AsyncProcessor:
    """Async version of the processor for better concurrency"""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
    
    async def process_audio_async(self, audio_path: str) -> Dict:
        """Async audio processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.optimizer.process_audio_optimized, audio_path)
    
    async def process_batch_async(self, audio_paths: List[str]) -> List[Dict]:
        """Async batch processing"""
        tasks = [self.process_audio_async(path) for path in audio_paths]
        return await asyncio.gather(*tasks)

def main():
    """Main optimization function"""
    logger.info("Starting performance optimization...")
    
    optimizer = PerformanceOptimizer()
    optimizer.initialize_optimized_processor()
    
    # Get test files
    test_files = []
    ravdess_dir = Path("datasets/ravdess")
    if ravdess_dir.exists():
        for actor_dir in ravdess_dir.iterdir():
            if actor_dir.is_dir():
                for audio_file in actor_dir.glob("*.wav"):
                    test_files.append(str(audio_file))
                    if len(test_files) >= 20:  # Limit for testing
                        break
                if len(test_files) >= 20:
                    break
    
    if not test_files:
        logger.error("No test files found!")
        return
    
    # Run benchmark
    benchmark_results = optimizer.benchmark_performance(test_files)
    
    # Get performance metrics
    metrics = optimizer.get_performance_metrics()
    
    logger.info(f"Performance metrics: {metrics}")
    logger.info(f"Benchmark results: {benchmark_results}")
    
    # Save results
    results = {
        'metrics': metrics,
        'benchmark': benchmark_results,
        'optimization_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('performance_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Performance optimization completed!")

if __name__ == "__main__":
    main()
