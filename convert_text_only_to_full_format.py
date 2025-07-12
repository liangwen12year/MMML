#!/usr/bin/env python3
"""
Convert text-only CSV to full 3-column format for sentiment analysis
"""

import pandas as pd
import sys
from tqdm import tqdm

def convert_text_only_to_full_format(input_file, output_file):
    print("🔄 CONVERTING TEXT-ONLY CSV TO FULL FORMAT")
    print("=" * 60)
    
    # Read the text-only CSV with robust parsing
    print(f"📖 Reading {input_file}...")
    
    # Try multiple methods to read the CSV
    read_methods = [
        ("Default", {}),
        ("Skip bad lines", {"on_bad_lines": 'skip'}),
        ("Manual parsing", {"sep": ',', "quotechar": '"', "doublequote": True, "skipinitialspace": True})
    ]
    
    df = None
    for method_name, kwargs in read_methods:
        try:
            df = pd.read_csv(input_file, **kwargs)
            if method_name != "Default":
                print(f"✅ Successfully read CSV using method: {method_name}")
            break
        except Exception as e:
            if method_name == "Default":
                print(f"❌ CSV parsing error: {e}")
                print("🔧 Trying with different CSV settings...")
            continue
    
    if df is None:
        print("❌ Could not read the CSV file with any method")
        return
    
    print(f"📊 Total lines: {len(df):,}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Handle different CSV structures
    if 'text' in df.columns and len(df.columns) == 1:
        # Perfect case: single text column
        text_col = 'text'
        print(f"📝 Found single 'text' column")
    elif len(df.columns) == 1:
        # Single column with different name
        text_col = df.columns[0]
        print(f"📝 Found single column: '{text_col}'")
    else:
        # Multiple columns - likely due to unquoted commas in text
        print(f"⚠️  Found {len(df.columns)} columns due to unquoted commas in text")
        print(f"📝 Columns: {list(df.columns)}")
        print("🔧 Merging all columns back into single text field...")
        
        # Merge all columns into a single text field
        df['merged_text'] = df.astype(str).agg(','.join, axis=1)
        text_col = 'merged_text'
        
        # Clean up the merged text (remove 'nan' values)
        df[text_col] = df[text_col].str.replace(',nan', '', regex=False)
        df[text_col] = df[text_col].str.replace('nan,', '', regex=False)
        df[text_col] = df[text_col].str.replace('nan', '', regex=False)
    
    print(f"📝 Using column '{text_col}' as text content")
    
    # Create the 3-column format
    print("🔧 Creating 3-column format...")
    
    # Generate conversation_id and utterance_id
    # For simplicity, treat all as one conversation with sequential utterance IDs
    conversation_id = 1
    utterance_ids = range(1, len(df) + 1)
    
    # Create new DataFrame
    new_df = pd.DataFrame({
        'conversation_id': [conversation_id] * len(df),
        'utterance_id': utterance_ids,
        'text': df[text_col]
    })
    
    # Handle any formatting issues
    print("🔧 Cleaning and formatting data...")
    
    # Remove any completely empty rows
    original_count = len(new_df)
    new_df = new_df.dropna(subset=['text'])
    new_df = new_df[new_df['text'].astype(str).str.strip() != '']
    cleaned_count = len(new_df)
    
    if original_count != cleaned_count:
        print(f"🗑️  Removed {original_count - cleaned_count} empty text entries")
    
    # Save the converted file
    print(f"💾 Saving to {output_file}...")
    new_df.to_csv(output_file, index=False)
    
    print(f"\n✅ CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"📊 Input: {original_count:,} text entries")
    print(f"📊 Output: {cleaned_count:,} data rows")
    print(f"📋 Format: conversation_id, utterance_id, text")
    print(f"💾 Saved to: {output_file}")
    
    # Show preview
    print(f"\n📖 Preview of converted data:")
    print(new_df.head(3).to_string(index=False))
    
    print(f"\n🚀 Ready for sentiment analysis!")
    print(f"💡 Run: python run_pretrained_sentiment.py --csv_path {output_file}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = "raw_data_text_only.csv"
        output_file = "raw_data_text_only_converted.csv"
    
    convert_text_only_to_full_format(input_file, output_file) 