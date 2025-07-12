#!/usr/bin/env python3
"""
Simple script to run 7-class sentiment analysis using pre-trained models
No training required - works out of the box!
"""

import os
import sys
import argparse
from pretrained_sentiment_inference import run_pretrained_inference

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run 7-class sentiment analysis using pre-trained models')
    parser.add_argument('--csv_path', type=str, default='sample_transcript.csv', help='Path to input CSV file')
    parser.add_argument('--output_path', type=str, default='sentiment_results_7class.csv', help='Path to output CSV file')
    parser.add_argument('--model_name', type=str, default='cardiffnlp/twitter-roberta-base-sentiment-latest', help='HuggingFace model name')
    parser.add_argument('--text_context_len', type=int, default=2, help='Number of previous utterances for context')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Configuration
    csv_path = args.csv_path
    output_path = args.output_path
    
    # Model options (choose one):
    # Option 1: Twitter-trained RoBERTa (good for informal text)
    model_name = args.model_name
    
    # Option 2: General domain RoBERTa (good for formal text)
    # model_name = "siebert/sentiment-roberta-large-english"
    
    # Option 3: DistilBERT (faster, good balance)
    # model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Parameters
    text_context_len = args.text_context_len  # Number of previous utterances to use as context
    batch_size = args.batch_size       # Batch size for inference
    
    # Check if input file exists
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file '{csv_path}' not found!")
        print("Please create your transcript CSV with columns: conversation_id, utterance_id, text")
        print("You can use the sample_transcript.csv as a template.")
        return
    
    print("🚀 Starting 7-Class Sentiment Analysis (CMU-MOSI/MOSEI Format)")
    print("=" * 60)
    print(f"📄 Input CSV: {csv_path}")
    print(f"🤖 Model: {model_name}")
    print(f"📊 Output: {output_path}")
    print(f"🔄 Context length: {text_context_len}")
    print(f"📦 Batch size: {batch_size}")
    print("=" * 60)
    
    # Run inference
    try:
        results_df = run_pretrained_inference(
            csv_path=csv_path,
            output_path=output_path,
            model_name=model_name,
            text_context_len=text_context_len,
            batch_size=batch_size
        )
        
        print("\n🎉 SUCCESS! Sentiment analysis completed.")
        print(f"📋 Results saved to: {output_path}")
        
        # Show detailed breakdown
        print("\n📈 Detailed Results:")
        print(results_df.groupby(['sentiment_class', 'sentiment_label']).size().reset_index(name='count').to_string(index=False))
        
        # Show some examples
        print("\n🔍 Sample Results:")
        sample_results = results_df[['conversation_id', 'utterance_id', 'original_text', 'sentiment_label', 'continuous_score']].head(5)
        for _, row in sample_results.iterrows():
            print(f"  {row['sentiment_label']:>18} ({row['continuous_score']:+.2f}): {row['original_text'][:80]}...")
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 