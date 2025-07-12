import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from inference_dataset import InferenceDataset
import warnings
import html
import urllib.parse

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def decode_text(text):
    """
    Decode text that contains HTML/URL encoded characters
    """
    if not isinstance(text, str):
        return text
    
    # Common encodings found in the data
    replacements = {
        '+ACI-': '"',      # quotes
        '+ACE-': '!',      # exclamation mark
        '+AH4-': '>',      # greater than
        '+AC0-': '-',      # hyphen
        '+AF0-': '_',      # underscore
        '+AFs-': '[',      # opening bracket
        '+AF0-': ']',      # closing bracket
        '+AHsAew-URL+AH0AfQ-': '{{URL}}',  # URL placeholder
        '+AHsAew-EMAIL+AH0AfQ-': '{{EMAIL}}',  # Email placeholder
        '+AHsAew-': '{{',   # opening curly brace
        '+AH0AfQ-': '}}',   # closing curly brace
        '+AHs-': '{',      # opening curly brace
        '+AH0-': '}',      # closing curly brace
        '+ACU-': '%',      # percent
        '+ACoAIg-': ':"',   # colon quote
        '+ACoAIQ-': ':"',   # colon quote variant
        '+AIQ-': '"',      # quote variant
        '+AIG-': '"',      # quote variant
        '+AD0-': '=',      # equals
        '+ACM-': '#',      # hash
        '+ACY-': '&',      # ampersand
        '+ADs-': ';',      # semicolon
        '+ACo-': '*',      # asterisk
        '+AD8-': '?',      # question mark
        '+AFs-scrubbed+AF0-': '[redacted]',  # scrubbed content
    }
    
    # Apply replacements
    decoded = text
    for encoded, replacement in replacements.items():
        decoded = decoded.replace(encoded, replacement)
    
    # Handle any remaining + patterns by trying URL decode
    try:
        if '+' in decoded:
            decoded = urllib.parse.unquote_plus(decoded)
    except:
        pass
    
    # Try HTML decode as fallback
    try:
        decoded = html.unescape(decoded)
    except:
        pass
    
    return decoded.strip()

class PretrainedSentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize with a pre-trained sentiment model
        
        Args:
            model_name: HuggingFace model name for sentiment analysis
        """
        print(f"Loading pre-trained model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Create sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None  # Return all scores (replaces deprecated return_all_scores)
        )
        
    def predict_sentiment_scores(self, texts):
        """
        Predict sentiment scores for a list of texts
        
        Returns:
            List of sentiment scores in range [-3, 3]
        """
        results = self.sentiment_pipeline(texts)
        sentiment_scores = []
        
        for result in results:
            # Extract scores for different labels
            # result is already a list of dicts with label and score
            scores = {item['label']: item['score'] for item in result}
            
            # Convert to continuous scale [-3, 3]
            # Handle different label formats (lowercase or uppercase)
            if 'negative' in scores and 'positive' in scores:
                neg_score = scores['negative']
                pos_score = scores['positive']
                neu_score = scores.get('neutral', 0.0)
                
                # Convert to [-3, 3] scale based on confidence
                if pos_score > max(neg_score, neu_score):
                    # Positive sentiment
                    confidence = pos_score - max(neg_score, neu_score)
                    sentiment_score = confidence * 3.0
                elif neg_score > max(pos_score, neu_score):
                    # Negative sentiment
                    confidence = neg_score - max(pos_score, neu_score)
                    sentiment_score = -confidence * 3.0
                else:
                    # Neutral sentiment
                    sentiment_score = 0.0
                    
            elif 'NEGATIVE' in scores and 'POSITIVE' in scores:
                neg_score = scores['NEGATIVE']
                pos_score = scores['POSITIVE']
                neu_score = scores.get('NEUTRAL', 0.0)
                
                # Convert to [-3, 3] scale
                if pos_score > max(neg_score, neu_score):
                    confidence = pos_score - max(neg_score, neu_score)
                    sentiment_score = confidence * 3.0
                elif neg_score > max(pos_score, neu_score):
                    confidence = neg_score - max(pos_score, neu_score)
                    sentiment_score = -confidence * 3.0
                else:
                    sentiment_score = 0.0
                    
            elif 'LABEL_0' in scores and 'LABEL_1' in scores and 'LABEL_2' in scores:
                # Handle 3-class models (negative, neutral, positive)
                neg_score = scores['LABEL_0']  # negative
                neu_score = scores['LABEL_1']  # neutral
                pos_score = scores['LABEL_2']  # positive
                
                # Map to [-3, 3] scale
                if pos_score > max(neg_score, neu_score):
                    sentiment_score = (pos_score - neu_score) * 3.0
                elif neg_score > max(pos_score, neu_score):
                    sentiment_score = -(neg_score - neu_score) * 3.0
                else:
                    sentiment_score = 0.0
            else:
                # Default fallback
                sentiment_score = 0.0
            
            # Clip to [-3, 3] range
            sentiment_score = np.clip(sentiment_score, -3.0, 3.0)
            sentiment_scores.append(sentiment_score)
        
        return sentiment_scores

def convert_to_7_classes(predictions):
    """
    Convert continuous predictions to 7-class labels
    Following CMU-MOSI/MOSEI evaluation format
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
    Get human-readable class names for 7-class sentiment (CMU-MOSI/MOSEI format)
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

class SimpleInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, text_context_length=2):
        """
        Simplified dataset for inference with context
        
        Supports two CSV formats:
        1. Single column: 'text' (generates conversation_id and utterance_id automatically)
        2. Three columns: 'conversation_id', 'utterance_id', 'text'
        """
        # Try multiple methods to read the CSV, preserving empty rows
        read_methods = [
            ("Default", {"keep_default_na": False, "na_filter": False}),
            ("With quotes", {"quotechar": '"', "escapechar": '\\', "keep_default_na": False, "na_filter": False}),
            ("Skip bad lines", {"on_bad_lines": 'skip', "keep_default_na": False, "na_filter": False}),
            ("Manual parsing", {"sep": ',', "quotechar": '"', "doublequote": True, "skipinitialspace": True, "keep_default_na": False, "na_filter": False})
        ]
        
        success = False
        for method_name, kwargs in read_methods:
            try:
                self.df = pd.read_csv(csv_path, **kwargs)
                if method_name != "Default":
                    print(f"✅ Successfully read CSV using method: {method_name}")
                success = True
                break
            except Exception as e:
                if method_name == "Default":
                    print(f"❌ CSV parsing error: {e}")
                    print("🔧 Trying with different CSV settings...")
                continue
        
        if not success:
            print(f"❌ Unable to read CSV file with any method.")
            print(f"Expected format: conversation_id,utterance_id,text")
            print(f"Make sure text is properly quoted if it contains commas.")
            print(f"Example: meeting_1,1,\"Hello, how are you?\"")
            raise ValueError("Could not read CSV file")
        
        # Fix column names if they're encoded (e.g., +AF8- for underscore)
        original_columns = list(self.df.columns)
        self.df.columns = self.df.columns.str.replace('+AF8-', '_', regex=False)
        self.df.columns = self.df.columns.str.replace('+', '', regex=False)
        
        if original_columns != list(self.df.columns):
            print(f"🔧 Fixed encoded column names:")
            for old, new in zip(original_columns, self.df.columns):
                if old != new:
                    print(f"   {old} → {new}")
        
        # Handle different CSV formats: 3-column or text-only
        original_column_count = len(original_columns)  # Use original count before any processing
        if len(self.df.columns) == 1:
            # Single column - assume it's text-only format
            text_col = self.df.columns[0]
            print(f"📝 Detected text-only format with column: '{text_col}'")
            
            # Generate conversation_id and utterance_id automatically for ALL rows (including empty ones)
            conversation_id = 1
            utterance_ids = range(1, len(self.df) + 1)
            
            # Create the 3-column structure internally
            self.df = self.df.rename(columns={text_col: 'text'})
            self.df.insert(0, 'conversation_id', conversation_id)
            self.df.insert(1, 'utterance_id', utterance_ids)
            
            print(f"🔧 Generated conversation_id and utterance_id for {len(self.df)} rows (including empty rows)")
            
        else:
            # Multi-column format - use original validation logic
            required_mapping = {
                'conversation_id': ['conversation_id', 'conversationid', 'conv_id'],
                'utterance_id': ['utterance_id', 'utteranceid', 'utt_id', 'message_id'],
                'text': ['text', 'message', 'content', 'utterance']
            }
            
            column_mapping = {}
            missing_cols = []
            
            for required_col, possible_names in required_mapping.items():
                found = False
                for possible_name in possible_names:
                    if possible_name in self.df.columns:
                        column_mapping[required_col] = possible_name
                        found = True
                        break
                if not found:
                    missing_cols.append(required_col)
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                print(f"Available columns: {list(self.df.columns)}")
                print(f"Expected one of these for each required column:")
                for req_col, possible in required_mapping.items():
                    print(f"   {req_col}: {possible}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Rename columns if needed
            if column_mapping != {'conversation_id': 'conversation_id', 'utterance_id': 'utterance_id', 'text': 'text'}:
                rename_dict = {v: k for k, v in column_mapping.items()}
                self.df = self.df.rename(columns=rename_dict)
                print(f"🔄 Mapped columns: {column_mapping}")
            
            # Keep only the required columns
            self.df = self.df[['conversation_id', 'utterance_id', 'text']]
        
        # Decode text content (handle encoded characters)
        print("🔧 Decoding text content...")
        self.df['text'] = self.df['text'].apply(decode_text)
        
        # Keep track of original row positions for line preservation
        self.df['original_position'] = range(len(self.df))
        
        # Mark empty rows but don't remove them yet
        # Check for various empty conditions more robustly
        self.df['is_empty'] = (
            (self.df['text'].isna()) | 
            (self.df['text'].astype(str).str.strip() == '') | 
            (self.df['text'].astype(str) == 'nan') |
            (self.df['text'].astype(str) == 'NaN') |
            (self.df['text'].astype(str) == 'None') |
            (self.df['text'].astype(str).str.strip() == 'nan') |
            (self.df['text'].astype(str).str.strip() == 'NaN') |
            (self.df['text'].astype(str).str.strip() == 'None')
        )
        
        print(f"📊 Total rows: {len(self.df)}")
        print(f"📝 Non-empty text rows: {(~self.df['is_empty']).sum()}")
        print(f"📄 Empty rows: {self.df['is_empty'].sum()}")
        
        # Don't sort if we want to preserve original line order for text-only format
        if original_column_count > 1:  # Multi-column format - sort by conversation and utterance
            self.df = self.df.sort_values(by=['conversation_id', 'utterance_id']).reset_index(drop=True)
        else:
            # Text-only format - preserve original order
            self.df = self.df.reset_index(drop=True)
        
        non_empty_count = (~self.df['is_empty']).sum()
        conversation_count = self.df[~self.df['is_empty']]['conversation_id'].nunique()
        print(f"📊 Loaded {non_empty_count} utterances from {conversation_count} conversations")
        
        # Apply English preprocessing (first char uppercase, rest lowercase)
        # Only do this if the text is not empty
        def preprocess_text(text):
            if not text or len(str(text).strip()) == 0 or str(text).strip() in ['nan', 'NaN', 'None']:
                return ""
            text_str = str(text).strip()
            if len(text_str) == 0:
                return ""
            return text_str[0].upper() + text_str[1:].lower() if len(text_str) > 1 else text_str.upper()
        
        self.df['text'] = self.df['text'].apply(preprocess_text)
        
        # Re-check for empty rows after preprocessing
        self.df['is_empty'] = (
            (self.df['text'].isna()) | 
            (self.df['text'].astype(str).str.strip() == '') | 
            (self.df['text'].astype(str) == 'nan') |
            (self.df['text'].astype(str) == 'NaN') |
            (self.df['text'].astype(str) == 'None') |
            (self.df['text'].astype(str).str.strip() == 'nan') |
            (self.df['text'].astype(str).str.strip() == 'NaN') |
            (self.df['text'].astype(str).str.strip() == 'None')
        )
        
        self.text_context_length = text_context_length
        
    def __getitem__(self, index):
        # Check if this is an empty row
        if self.df.iloc[index]['is_empty']:
            return {
                "full_text": "",
                "original_text": "",
                "conversation_id": str(self.df.iloc[index]['conversation_id']),
                "utterance_id": str(self.df.iloc[index]['utterance_id']),
                "is_empty": True,
                "original_position": self.df.iloc[index]['original_position']
            }
        
        # Current utterance
        text = str(self.df.iloc[index]['text'])
        
        # Build context from previous utterances in same conversation
        text_context = ''
        current_conv_id = self.df.iloc[index]['conversation_id']
        
        context_texts = []
        for i in range(1, self.text_context_length + 1):
            if index - i < 0:
                break
            # Skip empty rows when building context
            if self.df.iloc[index - i]['is_empty']:
                continue
            prev_conv_id = self.df.iloc[index - i]['conversation_id']
            if prev_conv_id != current_conv_id:
                break
            context = str(self.df.iloc[index - i]['text'])
            context_texts.append(context)
        
        # Combine context and current text
        if context_texts:
            # Reverse to get chronological order
            context_texts.reverse()
            full_text = ' '.join(context_texts) + ' ' + text
        else:
            full_text = text
        
        return {
            "full_text": full_text,
            "original_text": text,
            "conversation_id": str(self.df.iloc[index]['conversation_id']),
            "utterance_id": str(self.df.iloc[index]['utterance_id']),
            "is_empty": False,
            "original_position": self.df.iloc[index]['original_position']
        }
    
    def __len__(self):
        return len(self.df)

def run_pretrained_inference(csv_path, output_path, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                           text_context_len=2, batch_size=16):
    """
    Run sentiment inference using pre-trained models
    
    Args:
        csv_path: Path to input CSV with transcript
        output_path: Path to save results CSV
        model_name: HuggingFace model name
        text_context_len: Number of previous utterances for context
        batch_size: Batch size for inference
    """
    
    # Load dataset
    print(f"Loading dataset from {csv_path}...")
    dataset = SimpleInferenceDataset(csv_path, text_context_length=text_context_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize sentiment analyzer
    analyzer = PretrainedSentimentAnalyzer(model_name)
    
    # Run inference
    print("Running sentiment analysis...")
    all_predictions = []
    all_metadata = []
    
    for batch in tqdm(dataloader):
        # Get texts for this batch
        texts = batch["full_text"]
        is_empty = batch.get("is_empty", [False] * len(texts))
        
        try:
            # Process only non-empty texts for sentiment analysis
            non_empty_texts = [text for i, text in enumerate(texts) if not is_empty[i]]
            
            if non_empty_texts:
                # Predict sentiment scores for non-empty texts
                sentiment_scores = analyzer.predict_sentiment_scores(non_empty_texts)
            else:
                sentiment_scores = []
            
            # Store metadata and predictions, preserving empty rows
            sentiment_idx = 0
            for i in range(len(batch["conversation_id"])):
                metadata = {
                    'conversation_id': batch["conversation_id"][i],
                    'utterance_id': batch["utterance_id"][i], 
                    'original_text': batch["original_text"][i],
                    'original_position': batch["original_position"][i] if "original_position" in batch else i,
                    'is_empty': is_empty[i] if i < len(is_empty) else False
                }
                
                if is_empty[i] if i < len(is_empty) else False:
                    # Empty row - add placeholder values
                    all_predictions.append(0.0)  # Neutral score for empty rows
                else:
                    # Non-empty row - add actual prediction
                    if sentiment_idx < len(sentiment_scores):
                        all_predictions.append(sentiment_scores[sentiment_idx])
                        sentiment_idx += 1
                    else:
                        all_predictions.append(0.0)  # Fallback
                
                all_metadata.append(metadata)
                
        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            # Add placeholder data for this batch to maintain line count
            for i in range(len(batch["conversation_id"])):
                all_predictions.append(0.0)
                all_metadata.append({
                    'conversation_id': batch["conversation_id"][i],
                    'utterance_id': batch["utterance_id"][i], 
                    'original_text': batch["original_text"][i],
                    'original_position': batch["original_position"][i] if "original_position" in batch else i,
                    'is_empty': True
                })
            continue
    
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
    
    # Sort by original position to preserve line order
    if 'original_position' in results_df.columns:
        results_df = results_df.sort_values('original_position').reset_index(drop=True)
        # Keep track of empty rows before dropping the column
        empty_row_count = results_df['is_empty'].sum() if 'is_empty' in results_df.columns else 0
        results_df = results_df.drop(columns=['original_position', 'is_empty'])  # Remove helper columns
    else:
        empty_row_count = 0
    
    # Double-check that we maintain the exact line count
    print(f"📊 Input file line count check...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        input_line_count = sum(1 for line in f)
    print(f"📊 Input file lines: {input_line_count}")
    print(f"📊 Output DataFrame lines: {len(results_df)}")
    if len(results_df) != input_line_count:
        print(f"⚠️  WARNING: Line count mismatch! Expected {input_line_count}, got {len(results_df)}")
    else:
        print(f"✅ Line count preserved perfectly!")
    
    # Save results
    print(f"Saving results to {output_path}...")
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS RESULTS (7-Class CMU-MOSI/MOSEI Format)")
    print("="*50)
    
    # Count sentiment distribution (exclude empty row placeholders)
    non_empty_results = results_df[results_df['original_text'].astype(str).str.strip() != '']
    if len(non_empty_results) > 0:
        print(non_empty_results['sentiment_label'].value_counts().sort_index())
    else:
        print(results_df['sentiment_label'].value_counts().sort_index())
    
    print(f"\nTotal lines in output: {len(results_df)}")
    if empty_row_count > 0:
        print(f"Non-empty utterances analyzed: {len(results_df) - empty_row_count}")
        print(f"Empty lines preserved: {empty_row_count}")
    print(f"Results saved to: {output_path}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Pre-trained Sentiment Analysis Inference (7-Class)')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to input CSV with transcript')
    parser.add_argument('--output_path', type=str, default='sentiment_results_7class.csv', help='Path to save results CSV')
    parser.add_argument('--model_name', type=str, default='cardiffnlp/twitter-roberta-base-sentiment-latest', 
                       help='HuggingFace model name for sentiment analysis')
    parser.add_argument('--text_context_len', type=int, default=2, help='Number of previous utterances for context')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PRE-TRAINED SENTIMENT ANALYSIS (7-Class CMU-MOSI/MOSEI Format)")
    print("="*60)
    print(f"Input CSV: {args.csv_path}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_path}")
    print(f"Context length: {args.text_context_len}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)
    
    # Run inference
    try:
        results_df = run_pretrained_inference(
            csv_path=args.csv_path,
            output_path=args.output_path,
            model_name=args.model_name,
            text_context_len=args.text_context_len,
            batch_size=args.batch_size
        )
        
        print("\n" + "="*50)
        print("SUCCESS! Sentiment analysis completed.")
        print("="*50)
        
        # Show sample results
        print("\nSample results:")
        print(results_df[['conversation_id', 'utterance_id', 'original_text', 'sentiment_label', 'continuous_score']].head(10).to_string())
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        raise

if __name__ == "__main__":
    main() 