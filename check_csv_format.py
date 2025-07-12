#!/usr/bin/env python3
"""
CSV Format Checker and Fixer for Sentiment Analysis

This tool helps diagnose and fix common CSV formatting issues.
"""

import pandas as pd
import sys
import argparse
import re

def check_csv_format(csv_path, show_sample=True):
    """
    Check CSV format and provide diagnostic information
    """
    print(f"🔍 Checking CSV format: {csv_path}")
    print("=" * 50)
    
    # First, let's look at the raw file
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]  # First 10 lines
            
        print("📝 First few lines of the file:")
        for i, line in enumerate(lines, 1):
            print(f"Line {i}: {line.strip()}")
            
        print("\n" + "=" * 50)
        
        # Count commas in each line
        print("🔢 Comma count analysis:")
        for i, line in enumerate(lines[:5], 1):
            comma_count = line.count(',')
            print(f"Line {i}: {comma_count} commas")
            
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Try different pandas reading methods
    methods = [
        ("Default", {}),
        ("With quotes", {"quotechar": '"'}),
        ("Skip bad lines", {"on_bad_lines": 'skip'}),
        ("Custom separator", {"sep": ',', "quotechar": '"', "escapechar": '\\'}),
    ]
    
    successful_method = None
    
    for method_name, kwargs in methods:
        try:
            print(f"🧪 Trying method: {method_name}")
            df = pd.read_csv(csv_path, **kwargs)
            print(f"✅ Success! Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            if show_sample and len(df) > 0:
                print("   Sample data:")
                print(df.head(3).to_string(index=False))
            
            successful_method = (method_name, kwargs, df)
            break
            
        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}...")
        
        print()
    
    if successful_method:
        method_name, kwargs, df = successful_method
        print(f"\n🎉 Successfully read CSV using method: {method_name}")
        
        # Check for required columns
        required_cols = ['conversation_id', 'utterance_id', 'text']
        available_cols = list(df.columns)
        missing_cols = [col for col in required_cols if col not in available_cols]
        
        if missing_cols:
            print(f"⚠️  Missing required columns: {missing_cols}")
            print(f"📋 Available columns: {available_cols}")
            print(f"✅ Required columns: {required_cols}")
        else:
            print(f"✅ All required columns present: {required_cols}")
            
        return True, successful_method
    else:
        print("\n❌ Could not read CSV with any method")
        return False, None

def suggest_fixes(csv_path):
    """
    Suggest fixes for common CSV issues
    """
    print("\n🛠️  SUGGESTED FIXES:")
    print("=" * 50)
    
    print("1. 🔤 Quote text fields that contain commas:")
    print('   ❌ Bad:  meeting_1,1,Hello, how are you?')
    print('   ✅ Good: meeting_1,1,"Hello, how are you?"')
    
    print("\n2. 📋 Ensure proper column headers:")
    print("   Required: conversation_id,utterance_id,text")
    
    print("\n3. 🧹 Remove extra commas or empty fields")
    
    print("\n4. 💾 Example of correct format:")
    print("conversation_id,utterance_id,text")
    print('meeting_1,1,"Good morning everyone."')
    print('meeting_1,2,"I\'m excited about this project."')
    print('meeting_1,3,"However, I have some concerns."')

def create_sample_csv(output_path="fixed_sample.csv"):
    """
    Create a properly formatted sample CSV
    """
    sample_data = [
        ['conversation_id', 'utterance_id', 'text'],
        ['meeting_1', 1, 'Good morning everyone, let\'s start today\'s meeting.'],
        ['meeting_1', 2, 'I\'m really excited about this new project we\'re discussing.'],
        ['meeting_1', 3, 'I have some concerns about the proposed timeline, though.'],
        ['meeting_1', 4, 'The budget constraints are quite challenging for this scope.'],
        ['meeting_1', 5, 'But I believe we can make this work with proper planning.'],
        ['interview_1', 1, 'Hello, thank you for coming in today.'],
        ['interview_1', 2, 'I\'m nervous, but excited about this opportunity.'],
    ]
    
    # Write properly quoted CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in sample_data:
            if isinstance(row[0], str) and row[0] == 'conversation_id':
                # Header row
                f.write(','.join(row) + '\n')
            else:
                # Data rows - quote the text field
                f.write(f'{row[0]},{row[1]},"{row[2]}"\n')
    
    print(f"✅ Created properly formatted sample CSV: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Check and diagnose CSV format issues')
    parser.add_argument('csv_path', help='Path to CSV file to check')
    parser.add_argument('--create-sample', action='store_true', help='Create a sample CSV file')
    parser.add_argument('--no-sample', action='store_true', help='Don\'t show sample data')
    
    args = parser.parse_args()
    
    # Check the CSV format
    success, method_info = check_csv_format(args.csv_path, show_sample=not args.no_sample)
    
    if not success:
        suggest_fixes(args.csv_path)
        
        if args.create_sample:
            create_sample_csv()
    else:
        print(f"\n🎉 Your CSV file looks good and can be processed!")
        if method_info:
            method_name, kwargs, df = method_info
            if kwargs:  # If special parameters were needed
                print(f"💡 Note: Your CSV required special handling ({method_name})")
                print("   The sentiment analysis script should handle this automatically.")
    
    if args.create_sample:
        create_sample_csv()

if __name__ == "__main__":
    main() 