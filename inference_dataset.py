import torch
import pandas as pd
from transformers import AutoTokenizer

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, text_context_length=2, language='en'):
        """
        Dataset for inference-only sentiment analysis
        
        Args:
            csv_path: Path to CSV with columns: conversation_id, utterance_id, text
            text_context_length: Number of previous utterances to use as context
            language: 'en' for English, 'zh' for Chinese
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sort_values(by=['conversation_id', 'utterance_id']).reset_index(drop=True)
        
        # Text preprocessing for English (like in original code)
        if language == 'en':
            self.df['text'] = self.df['text'].str[0] + self.df['text'].str[1:].apply(lambda x: x.lower())
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
            self.max_length = 96
        else:  # Chinese
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
            self.max_length = 64
        
        self.text_context_length = text_context_length
        
    def __getitem__(self, index):
        # Current utterance
        text = str(self.df.iloc[index]['text'])
        
        # Build context from previous utterances in same conversation
        text_context = ''
        current_conv_id = self.df.iloc[index]['conversation_id']
        
        for i in range(1, self.text_context_length + 1):
            if index - i < 0:
                break
            prev_conv_id = self.df.iloc[index - i]['conversation_id']
            if prev_conv_id != current_conv_id:
                break
            context = str(self.df.iloc[index - i]['text'])
            text_context = context + '</s>' + text_context
        
        # Remove trailing separator
        if text_context.endswith('</s>'):
            text_context = text_context[:-4]
        
        # Tokenize current utterance
        tokenized_text = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )
        
        # Tokenize context
        tokenized_context = self.tokenizer(
            text_context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )
        
        return {
            "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
            "text_context_tokens": torch.tensor(tokenized_context["input_ids"], dtype=torch.long),
            "text_context_masks": torch.tensor(tokenized_context["attention_mask"], dtype=torch.long),
            "conversation_id": self.df.iloc[index]['conversation_id'],
            "utterance_id": self.df.iloc[index]['utterance_id'],
            "original_text": text
        }
    
    def __len__(self):
        return len(self.df) 