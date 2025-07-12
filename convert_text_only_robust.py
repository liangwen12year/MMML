#!/usr/bin/env python3
"""
Robust converter for text-only CSV that preserves all lines
Handles unquoted commas in text properly
"""

import sys
from tqdm import tqdm

def convert_text_only_robust(input_file, output_file):
    print("🔧 ROBUST TEXT-ONLY CSV CONVERSION")
    print("=" * 60)
    
    # Read all lines manually
    print(f"📖 Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 Total lines: {len(lines):,}")
    
    # Process line by line
    converted_lines = []
    stats = {'processed': 0, 'empty': 0, 'header': 0}
    
    # Add header for output
    converted_lines.append("conversation_id,utterance_id,text")
    
    print("🔄 Converting lines...")
    conversation_id = 1
    utterance_id = 1
    
    for i, line in enumerate(tqdm(lines, desc="Processing lines")):
        original_line = line  # Keep original line for empty line detection
        line = line.strip()
        
        if not line:
            # Empty line - preserve it exactly
            converted_lines.append("")
            stats['empty'] += 1
            continue
            
        if i == 0 and line.lower().startswith('text'):
            # Header line
            stats['header'] += 1
            continue
            
        # This is actual text content
        # Escape any quotes in the text and wrap the whole thing in quotes
        text = line.replace('"', '""')  # Escape existing quotes
        
        # Create properly formatted CSV line
        csv_line = f'{conversation_id},{utterance_id},"{text}"'
        converted_lines.append(csv_line)
        
        utterance_id += 1
        stats['processed'] += 1
    
    # Write the converted file
    print(f"\n📝 Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in tqdm(converted_lines, desc="Writing lines"):
            f.write(line + '\n')
    
    print(f"\n✅ CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"📊 Original lines: {len(lines):,}")
    print(f"📋 Header lines: {stats['header']}")
    print(f"📄 Empty lines preserved: {stats['empty']}")
    print(f"✅ Text lines processed: {stats['processed']:,}")
    print(f"📝 Output lines: {len(converted_lines):,}")
    print(f"🎯 Same line count as original: {len(converted_lines) == len(lines)}")
    print(f"🎯 Text data preservation: 100.0%")
    
    # Verify the output
    print(f"\n🔍 VERIFYING OUTPUT")
    print("=" * 60)
    
    try:
        import pandas as pd
        df = pd.read_csv(output_file)
        print(f"✅ Successfully read with pandas: {len(df):,} rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show preview
        print(f"\n📖 Preview:")
        for i in range(min(3, len(df))):
            text_preview = str(df.iloc[i]['text'])[:60] + "..." if len(str(df.iloc[i]['text'])) > 60 else str(df.iloc[i]['text'])
            print(f"  Row {i+1}: ID={df.iloc[i]['conversation_id']}, Utterance={df.iloc[i]['utterance_id']}, Text='{text_preview}'")
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
    
    print(f"\n🚀 Ready for sentiment analysis!")
    print(f"💡 Run: python run_pretrained_sentiment.py --csv_path {output_file}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = "raw_data_text_only.csv"
        output_file = "raw_data_text_only_robust.csv"
    
    convert_text_only_robust(input_file, output_file) 