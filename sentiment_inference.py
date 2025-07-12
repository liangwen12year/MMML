import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from inference_dataset import InferenceDataset
from utils.context_model import roberta_en_context
from utils.en_train import EnConfig
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def convert_to_7_classes(predictions):
    """
    Convert continuous predictions to 7-class labels
    Following the same logic as in the original evaluation code
    """
    # Clip predictions to [-3, 3] range
    clipped_preds = np.clip(predictions, a_min=-3.0, a_max=3.0)
    
    # Round to get discrete classes
    class_labels = np.round(clipped_preds).astype(int)
    
    # Map to 0-6 range for class indices
    class_indices = class_labels + 3  # -3 becomes 0, +3 becomes 6
    
    return class_labels, class_indices

def get_class_names():
    """
    Get human-readable class names for 7-class sentiment
    """
    return {
        -3: "Highly Negative",
        -2: "Negative", 
        -1: "Slightly Negative",
        0: "Neutral",
        1: "Slightly Positive",
        2: "Positive",
        3: "Highly Positive"
    }

def run_inference(csv_path, model_path, output_path, text_context_len=2, batch_size=16, language='en'):
    """
    Run sentiment inference on transcript CSV
    
    Args:
        csv_path: Path to input CSV with transcript
        model_path: Path to trained model checkpoint
        output_path: Path to save results CSV
        text_context_len: Number of previous utterances for context
        batch_size: Batch size for inference
        language: 'en' for English, 'zh' for Chinese
    """
    
    # Load dataset
    print(f"Loading dataset from {csv_path}...")
    dataset = InferenceDataset(csv_path, text_context_length=text_context_len, language=language)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    print(f"Loading model from {model_path}...")
    if language == 'en':
        model = roberta_en_context()
    else:
        # For Chinese, you'd need to implement the Chinese version
        raise NotImplementedError("Chinese model not implemented in this script")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Run inference
    print("Running inference...")
    all_predictions = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            text_tokens = batch["text_tokens"].to(device)
            text_masks = batch["text_masks"].to(device)
            text_context_tokens = batch["text_context_tokens"].to(device)
            text_context_masks = batch["text_context_masks"].to(device)
            
            # Forward pass - text only model
            outputs = model(text_tokens, text_masks, text_context_tokens, text_context_masks)
            
            # Store predictions and metadata
            predictions = outputs.cpu().numpy()
            all_predictions.extend(predictions.flatten())
            
            # Store metadata
            for i in range(len(batch["conversation_id"])):
                all_metadata.append({
                    'conversation_id': batch["conversation_id"][i],
                    'utterance_id': batch["utterance_id"][i], 
                    'original_text': batch["original_text"][i]
                })
    
    # Convert to 7-class labels
    print("Converting to 7-class labels...")
    class_labels, class_indices = convert_to_7_classes(np.array(all_predictions))
    class_names_dict = get_class_names()
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_metadata)
    results_df['continuous_score'] = all_predictions
    results_df['sentiment_class'] = class_labels
    results_df['class_index'] = class_indices
    results_df['sentiment_label'] = results_df['sentiment_class'].map(class_names_dict)
    
    # Save results
    print(f"Saving results to {output_path}...")
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\nSentiment Analysis Summary:")
    print(results_df['sentiment_label'].value_counts().sort_index())
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Inference')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to input CSV with transcript')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--output_path', type=str, default='sentiment_results.csv', help='Path to save results CSV')
    parser.add_argument('--text_context_len', type=int, default=2, help='Number of previous utterances for context')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'], help='Language: en for English, zh for Chinese')
    
    args = parser.parse_args()
    
    # Run inference
    results_df = run_inference(
        csv_path=args.csv_path,
        model_path=args.model_path,
        output_path=args.output_path,
        text_context_len=args.text_context_len,
        batch_size=args.batch_size,
        language=args.language
    )
    
    print(f"\nInference completed! Results saved to {args.output_path}")

if __name__ == "__main__":
    main() 