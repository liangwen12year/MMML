#!/usr/bin/env python3
"""
Properly fix CSV formatting by quoting text fields that contain commas
"""

import csv
import sys
from tqdm import tqdm

def fix_csv_properly(input_file, output_file):
    print("🔧 PROPERLY FIXING CSV FORMATTING")
    print("=" * 60)
    
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📖 Processing {len(lines)} lines from {input_file}")
    
    # Get header
    header = lines[0].strip()
    print(f"📋 Header: {header}")
    
    # Process each line
    fixed_lines = []
    stats = {'correct': 0, 'fixed': 0, 'skipped': 0, 'empty': 0}
    
    for i, line in enumerate(tqdm(lines, desc="Processing lines")):
        original_line = line  # Keep original line with newlines
        line = line.strip()
        
        # Preserve empty lines exactly as they are
        if not line:
            fixed_lines.append("")  # Add empty line
            stats['empty'] += 1
            continue
            
        if i == 0:  # Header
            fixed_lines.append(line)
            continue
            
        # Split on commas to see column count
        parts = line.split(',')
        
        if len(parts) == 3:
            # Perfect - already correct format
            conversation_id, utterance_id, text = parts
            # Still quote the text field to be safe
            fixed_line = f'{conversation_id},{utterance_id},"{text}"'
            fixed_lines.append(fixed_line)
            stats['correct'] += 1
            
        elif len(parts) > 3:
            # Text contains unquoted commas - fix it
            conversation_id = parts[0]
            utterance_id = parts[1]
            
            # Everything after the second comma is part of the text
            text_parts = parts[2:]
            text = ','.join(text_parts)  # Rejoin with commas
            
            # Quote the text field properly
            fixed_line = f'{conversation_id},{utterance_id},"{text}"'
            fixed_lines.append(fixed_line)
            stats['fixed'] += 1
            
        elif len(parts) < 3:
            # Missing columns - pad with empty values
            conversation_id = parts[0] if len(parts) > 0 else ""
            utterance_id = parts[1] if len(parts) > 1 else ""
            text = parts[2] if len(parts) > 2 else ""
            
            fixed_line = f'{conversation_id},{utterance_id},"{text}"'
            fixed_lines.append(fixed_line)
            stats['fixed'] += 1
        else:
            # Shouldn't happen but skip if it does
            stats['skipped'] += 1
    
    # Write properly formatted CSV
    print(f"\n📝 Writing properly formatted CSV to {output_file}")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        for line in tqdm(fixed_lines, desc="Writing lines"):
            f.write(line + '\n')
    
    print(f"\n📊 RESULTS:")
    print(f"  ✅ Already correct: {stats['correct']} lines")
    print(f"  🔧 Fixed: {stats['fixed']} lines")
    print(f"  📄 Empty lines preserved: {stats['empty']} lines")
    print(f"  ❌ Skipped: {stats['skipped']} lines")
    print(f"  📝 Total output: {len(fixed_lines)} lines")
    print(f"  🎯 Same line count as original: {len(fixed_lines) == len(lines)}")
    
    # Verify the fixed file
    print(f"\n🔍 VERIFYING FIXED FILE")
    print("=" * 60)
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines_read = list(reader)
            
        print(f"✅ Successfully read {len(lines_read)} lines with csv.reader")
        print(f"📋 All lines have {len(lines_read[1])} columns (expected: 3)")
        
        # Check first few lines
        print(f"\n📖 First 3 data rows:")
        for i, row in enumerate(lines_read[1:4]):
            print(f"  Row {i+1}: {len(row)} columns")
            print(f"    ID: {row[0]}")
            print(f"    Utterance: {row[1]}")
            print(f"    Text: {row[2][:80]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error verifying: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = "raw_data_100.csv"
        output_file = "raw_data_100_properly_fixed.csv"
    
    fix_csv_properly(input_file, output_file) 