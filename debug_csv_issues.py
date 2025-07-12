#!/usr/bin/env python3
"""
Debug exactly which utterances are being lost in CSV parsing
"""

import pandas as pd
import csv
import re
import sys
from tqdm import tqdm

def debug_and_fix_csv(input_file, output_file):
    print("🔍 ANALYZING CSV FILE")
    print("=" * 60)
    
    # Read raw lines
    print(f"📖 Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    # Check header
    header_line = lines[0].strip()
    print(f"Header: {header_line}")
    
    # Detect column names (handle encoded names)
    if 'conversation+AF8-id' in header_line:
        print("🔧 Detected encoded column names")
        header_parts = header_line.split(',')
        conversation_col = 'conversation+AF8-id'
        utterance_col = 'utterance+AF8-id'
        text_col = 'text'
    else:
        header_parts = header_line.split(',')
        conversation_col = 'conversation_id'
        utterance_col = 'utterance_id' 
        text_col = 'text'
    
    print(f"Expected columns: {header_parts}")
    print()
    
    # Analyze each line and fix issues
    fixed_lines = [header_line]  # Start with header
    
    print("🔧 FIXING PROBLEMATIC LINES")
    print("=" * 60)
    
    # Process with progress bar for large files
    fixed_count = 0
    skipped_count = 0
    
    for i, line in enumerate(tqdm(lines[1:], desc="Processing lines"), 2):  # Start from line 2 (after header)
        line = line.strip()
        if not line:
            continue
            
        # Split by comma
        parts = line.split(',')
        
        # More lenient approach: focus on extracting text content
        if len(parts) >= 1:
            # Try to identify the structure regardless of column count
            fixed_line = None
            
            if len(parts) == 3:
                # Perfect case: conversation_id, utterance_id, text
                fixed_line = line
            elif len(parts) < 3:
                # Too few columns - pad with empty values
                conversation_id = parts[0] if len(parts) > 0 else ""
                utterance_id = parts[1] if len(parts) > 1 else ""
                text = parts[2] if len(parts) > 2 else ""
                fixed_line = f"{conversation_id},{utterance_id},{text}"
                if i <= 20:  # Only show first 20 fixes to avoid spam
                    print(f"✅ Fixed line {i}: padded missing columns")
            else:
                # Too many columns - merge text parts
                conversation_id = parts[0]
                utterance_id = parts[1]
                
                # Everything after the second comma is text
                text_parts = parts[2:]
                
                # Clean and merge text parts
                cleaned_text_parts = []
                for part in text_parts:
                    # Remove quotes if present
                    part = part.strip().strip('"').strip("'")
                    if part:  # Only add non-empty parts
                        cleaned_text_parts.append(part)
                
                # Join with comma and space
                merged_text = ', '.join(cleaned_text_parts)
                
                # Wrap in quotes if contains comma
                if ',' in merged_text:
                    merged_text = f'"{merged_text}"'
                
                fixed_line = f"{conversation_id},{utterance_id},{merged_text}"
                if i <= 20:  # Only show first 20 fixes to avoid spam
                    print(f"✅ Fixed line {i}: merged {len(text_parts)} text parts")
                fixed_count += 1
            
            if fixed_line:
                fixed_lines.append(fixed_line)
        else:
            if i <= 20:  # Only show first 20 skips to avoid spam
                print(f"❌ Skipped line {i}: completely empty")
            skipped_count += 1
    
    # Write fixed CSV
    print(f"\n📝 WRITING FIXED CSV: {output_file}")
    print("=" * 60)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in tqdm(fixed_lines, desc="Writing lines"):
            f.write(line + '\n')
    
    print(f"✅ Fixed CSV written with {len(fixed_lines)} lines (including header)")
    print(f"📊 Data rows: {len(fixed_lines) - 1}")
    print(f"🔧 Lines fixed: {fixed_count}")
    print(f"❌ Lines skipped: {skipped_count}")
    print(f"📈 Success rate: {(len(fixed_lines) - 1) / (len(lines) - 1) * 100:.1f}%")
    
    # Verify the fixed file can be read
    print(f"\n🔍 VERIFYING FIXED FILE")
    print("=" * 60)
    
    try:
        print("📖 Testing pandas read...")
        df = pd.read_csv(output_file, nrows=1000)  # Test with first 1000 rows
        print(f"✅ Successfully loaded first 1000 rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Count non-empty text entries
        text_column = df.columns[-1]  # Assume last column is text
        non_empty_text = df[text_column].notna().sum()
        print(f"📝 Non-empty text entries in sample: {non_empty_text}")
        
        # Show some examples
        print(f"\n📖 First 3 text entries:")
        for i, text in enumerate(df[text_column].head(3)):
            print(f"  {i+1}: {str(text)[:100]}...")
            
    except Exception as e:
        print(f"❌ Error reading fixed file: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        # Default to raw_data.csv if no arguments provided
        input_file = "raw_data.csv"
        output_file = "raw_data_fixed.csv"
    
    debug_and_fix_csv(input_file, output_file) 