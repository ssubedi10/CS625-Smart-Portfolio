#!/usr/bin/env python3
"""
Test script for cache cleaning functionality
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_portfolio_analyzer.data_manager import DataManager

def test_cache_cleaning():
    """Test the cache cleaning functionality"""
    
    # Initialize DataManager (you'll need to provide a valid API key)
    try:
        # Try to get API key from environment
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('POLYGON_API_KEY')
        
        if not api_key:
            print("Warning: No POLYGON_API_KEY found. Creating DataManager without API key for testing...")
            # Create a dummy DataManager for testing cache functionality only
            data_manager = DataManager.__new__(DataManager)
            data_manager.cache_dir = os.path.join(os.path.dirname(__file__), 'data_cache')
            os.makedirs(data_manager.cache_dir, exist_ok=True)
        else:
            data_manager = DataManager(api_key=api_key)
            
    except Exception as e:
        print(f"Error initializing DataManager: {e}")
        return
    
    print("=== Cache Management Test ===")
    
    # Show current cache info
    print("\n1. Current cache information:")
    cache_info = data_manager.get_cache_info()
    
    if 'error' in cache_info:
        print(f"Error: {cache_info['error']}")
        return
    
    print(f"   Total files: {cache_info['total_files']}")
    print(f"   Total size: {cache_info['total_size_mb']} MB")
    print(f"   Oldest file: {cache_info['oldest_file']}")
    print(f"   Newest file: {cache_info['newest_file']}")
    
    print("\n   Files by age:")
    age_dist = cache_info['files_by_age']
    print(f"   • Less than 1 day: {age_dist.get('less_than_1_day', 0)}")
    print(f"   • 1-7 days: {age_dist.get('1_to_7_days', 0)}")
    print(f"   • 7-30 days: {age_dist.get('7_to_30_days', 0)}")
    print(f"   • Older than 30 days: {age_dist.get('older_than_30_days', 0)}")
    
    # Test cache cleaning with different thresholds
    test_days = [7, 30, 90]
    
    for days in test_days:
        print(f"\n2. Testing cache cleanup (keep last {days} days):")
        
        # Count files that would be deleted
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        files_to_delete = 0
        
        if os.path.exists(data_manager.cache_dir):
            for filename in os.listdir(data_manager.cache_dir):
                file_path = os.path.join(data_manager.cache_dir, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time < cutoff_time:
                        files_to_delete += 1
        
        print(f"   Files that would be deleted: {files_to_delete}")
        
        # Ask user if they want to proceed
        if files_to_delete > 0:
            response = input(f"   Delete {files_to_delete} files older than {days} days? (y/n): ")
            if response.lower() == 'y':
                deleted = data_manager.clean_old_cache(days_to_keep=days)
                print(f"   ✓ Actually deleted: {deleted} files")
            else:
                print("   Skipped cleanup")
        else:
            print("   No files to delete")
    
    # Show final cache info
    print("\n3. Final cache information:")
    final_cache_info = data_manager.get_cache_info()
    print(f"   Total files: {final_cache_info['total_files']}")
    print(f"   Total size: {final_cache_info['total_size_mb']} MB")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_cache_cleaning()
