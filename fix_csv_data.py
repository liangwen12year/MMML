#!/usr/bin/env python3
"""
Fix CSV formatting issues for sentiment analysis
"""

import pandas as pd
import re
import sys

def fix_csv_format(input_path, output_path):
    """
    Fix common CSV issues and save a clean version
    """
    print(f"🔧 Fixing CSV format: {input_path}")
    
    # Try to read with different methods
    df = None
    
    # Method 1: Skip bad lines first
    try:
        df = pd.read_csv(input_path, on_bad_lines='skip')
        print(f"✅ Read CSV with {len(df)} rows (skipped some problematic lines)")
    except:
        try:
            # Method 2: Try with quote handling
            df = pd.read_csv(input_path, quotechar='"', escapechar='\\')
            print(f"✅ Read CSV with quote handling: {len(df)} rows")
        except Exception as e:
            print(f"❌ Could not read CSV: {e}")
            return False
    
    if df is None:
        print("❌ Failed to read CSV")
        return False
    
    print(f"📋 Original columns: {list(df.columns)}")
    
    # Fix column names - decode the +AF8- encoding
    df.columns = df.columns.str.replace('+AF8-', '_', regex=False)
    df.columns = df.columns.str.replace('+', '', regex=False)  # Remove any other + characters
    
    print(f"🔧 Fixed columns: {list(df.columns)}")
    
    # Ensure we have the required columns
    required_mapping = {
        'conversation_id': ['conversation_id', 'conversationid', 'conv_id'],
        'utterance_id': ['utterance_id', 'utteranceid', 'utt_id', 'message_id'],
        'text': ['text', 'message', 'content', 'utterance']
    }
    
    # Try to map columns
    final_columns = {}
    for required_col, possible_names in required_mapping.items():
        found = False
        for possible_name in possible_names:
            if possible_name in df.columns:
                final_columns[required_col] = possible_name
                found = True
                break
        if not found:
            print(f"❌ Could not find column for {required_col}")
            print(f"   Available columns: {list(df.columns)}")
            return False
    
    # Create clean dataframe with proper column names
    clean_df = pd.DataFrame()
    clean_df['conversation_id'] = df[final_columns['conversation_id']]
    clean_df['utterance_id'] = df[final_columns['utterance_id']]
    clean_df['text'] = df[final_columns['text']]
    
    # Clean the data
    # Remove rows with missing text
    clean_df = clean_df.dropna(subset=['text'])
    
    # Remove empty text
    clean_df = clean_df[clean_df['text'].str.strip() != '']
    
    # Sort by conversation and utterance
    clean_df = clean_df.sort_values(['conversation_id', 'utterance_id']).reset_index(drop=True)
    
    print(f"🧹 Cleaned data: {len(clean_df)} rows")
    print(f"📊 Conversations: {clean_df['conversation_id'].nunique()}")
    
    # Save the clean CSV with proper quoting
    clean_df.to_csv(output_path, index=False, quoting=1)  # quoting=1 means quote all fields
    print(f"💾 Saved clean CSV to: {output_path}")
    
    # Show sample
    print("\n📝 Sample of cleaned data:")
    print(clean_df.head())
    
    return True

def try_recover_skipped_lines(input_path):
    """
    Try to recover lines that were skipped
    """
    print(f"\n🔍 Analyzing skipped lines in: {input_path}")
    
    # Read the raw file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📄 Total lines in file: {len(lines)}")
    
    # Check each line for comma count
    header_comma_count = lines[0].count(',') if lines else 0
    problematic_lines = []
    
    for i, line in enumerate(lines[1:], 2):  # Start from line 2 (after header)
        comma_count = line.count(',')
        if comma_count != header_comma_count:
            problematic_lines.append((i, comma_count, line.strip()))
    
    if problematic_lines:
        print(f"⚠️  Found {len(problematic_lines)} problematic lines:")
        for line_num, comma_count, content in problematic_lines[:5]:  # Show first 5
            print(f"   Line {line_num}: {comma_count} commas (expected {header_comma_count})")
            print(f"      Content: {content[:100]}...")
    else:
        print("✅ No obviously problematic lines found")
    
    return problematic_lines

def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_csv_data.py input.csv output.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("🚀 CSV Fixer Tool")
    print("=" * 50)
    
    # Analyze problematic lines
    try_recover_skipped_lines(input_path)
    
    # Fix the CSV format
    success = fix_csv_format(input_path, output_path)
    
    if success:
        print(f"\n🎉 SUCCESS! Clean CSV saved to: {output_path}")
        print(f"💡 Now run sentiment analysis with the fixed file:")
        print(f"   python pretrained_sentiment_inference.py --csv_path {output_path}")
    else:
        print(f"\n❌ Failed to fix CSV format")

if __name__ == "__main__":
    main() 