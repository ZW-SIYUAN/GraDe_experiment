import random
import typing as tp

import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class GraDeDataset(Dataset):
    """Dynamic Graph Dataset for tabular data

    Enhanced dataset class that handles both permutation of tabular data features
    and tracks feature-to-token mappings to support functional dependency constraints.
    """

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for processing text

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

    def _getitem(self, key, decoded=True, **kwargs):
        """Get item from tabular data with feature-to-token mapping

        Gets one instance of tabular data, randomly permutes the columns,
        converts to text, tokenizes, and tracks feature-to-token mappings.

        Args:
            key: Index of the item to retrieve
            decoded: Whether to return decoded items
            kwargs: Additional arguments

        Returns:
            Dictionary containing tokenized text and feature-to-token mappings
        """
        # Get original row data
        row = self._data.fast_slice(key, 1)
        
        # Randomly shuffle column order
        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)
        
        # Build text and record feature positions
        full_text = ""
        features_info = []  # (feature_name, start_position)
        
        for i, idx in enumerate(shuffle_idx):
            # Get feature name and value
            feat_name = row.column_names[idx]
            feat_value = str(row.columns[idx].to_pylist()[0]).strip()
            
            # Build feature segment
            prefix = ", " if i > 0 else ""
            feat_text = f"{prefix}{feat_name} is {feat_value}"
            
            # Record feature's starting position in text
            start_pos = len(full_text)
            full_text += feat_text
            
            # Save feature info
            features_info.append((feat_name, start_pos))
        
        # Tokenize the complete text
        tokenized_text = self.tokenizer(full_text, padding=True)
        
        # Build feature-to-token mapping
        feature_to_tokens = {}
        
        # For each feature, use its position in the text to find corresponding tokens
        for feat_name, start_pos in features_info:
            # Get token index for this feature
            start_token = self.tokenizer.char_to_token(start_pos) if hasattr(self.tokenizer, 'char_to_token') else None
            
            if start_token is not None:
                # Find tokens after the feature name and "is"
                feature_to_tokens[feat_name] = []
                
                # Simple method: find all tokens until next comma or end of text
                i = start_token
                while i < len(tokenized_text["input_ids"]):
                    feature_to_tokens[feat_name].append(i)
                    i += 1
                    # Stop if we find a comma
                    if i < len(tokenized_text["input_ids"]) and tokenized_text["input_ids"][i] == self.tokenizer.convert_tokens_to_ids([","])[0]:
                        break
        
        # Add feature-to-token mapping to result
        tokenized_text["feature_to_tokens"] = feature_to_tokens
        
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


@dataclass
class GraDeDataCollator(DataCollatorWithPadding):
    """Data collator for dynamic graph learning with tabular data

    Handles padding of input_ids and labels, and preserves feature-to-token mappings
    to support functional dependency constraints during training.
    """

    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        # Extract and save feature_to_tokens mappings
        feature_to_tokens_list = []
        for feature in features:
            # Make a copy to avoid modifying original features
            feature_to_tokens = feature.pop("feature_to_tokens", {})
            feature_to_tokens_list.append(feature_to_tokens)
        
        # Process other features with parent class methods
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        
        # Add feature_to_tokens to batch
        batch["feature_to_tokens"] = feature_to_tokens_list
        
        return batch 