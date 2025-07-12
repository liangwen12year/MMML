#!/usr/bin/env python3
"""
Simple example script for running sentiment analysis on transcript data
"""

import os
import sys
from sentiment_inference import run_inference

def main():
    # Configuration
    csv_path = "sample_transcript.csv"  # Your transcript CSV file
    model_path = "checkpoint/text_model.pth"  # Path to your trained model
    output_path = "sentiment_results.csv"  # Output file
    
    # Parameters
    text_context_len = 2  # Number of previous utterances to use as context
    batch_size = 16       # Batch size for inference
    language = 'en'       # Language: 'en' for English, 'zh' for Chinese
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file '{csv_path}' not found!")
        print("Please create your transcript CSV with columns: conversation_id, utterance_id, text")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train a model first or download a pre-trained model.")
        print("To train a text-only model, run:")
        print("python run.py --tasks T --dataset mosi --context True --text_context_len 2")
        return
    
    print("Starting sentiment analysis...")
    print(f"Input CSV: {csv_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Context length: {text_context_len}")
    print(f"Language: {language}")
    print("-" * 50)
    
    # Run inference
    try:
        results_df = run_inference(
            csv_path=csv_path,
            model_path=model_path,
            output_path=output_path,
            text_context_len=text_context_len,
            batch_size=batch_size,
            language=language
        )
        
        print("\nSuccess! Sentiment analysis completed.")
        print(f"Results saved to: {output_path}")
        print("\nSample results:")
        print(results_df[['conversation_id', 'utterance_id', 'original_text', 'sentiment_label', 'continuous_score']].head(10))
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return

if __name__ == "__main__":
    main() 