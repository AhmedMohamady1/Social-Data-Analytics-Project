"""
Bing Liu Dictionary-Based Sentiment Analysis with Negation Handling
====================================================================

This script implements a dictionary-based sentiment analysis model using the Bing Liu
positive and negative word lists. It includes sophisticated negation handling to account
for words like "not", "no", "without", etc. that can reverse sentiment polarity.

Author: SDA Project
Date: 2026
License: MIT

Features:
- Negation window handling (checks words within a specified window before sentiment words)
- Intensifier support (multiplies sentiment scores)
- Sentiment smoothing (considers overall text balance)
- Detailed sentiment analysis with scores
- Batch processing for multiple CSV files
- Flexible output with customizable sentiment labels
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, Tuple, List, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BingLiuSentimentAnalyzer:
    """
    Sentiment analyzer using Bing Liu dictionaries with negation handling.
    
    This class provides sophisticated sentiment analysis incorporating:
    - Positive and negative word dictionaries
    - Negation word detection
    - Intensifiers (very, extremely, etc.)
    - Sentiment polarity reversal for negated phrases
    """
    
    def __init__(self, positive_dict_path: str, negative_dict_path: str, 
                 negation_window: int = 3, intensifiers: Set[str] = None):
        """
        Initialize the sentiment analyzer.
        
        Parameters:
        -----------
        positive_dict_path : str
            Path to the file containing positive words (one per line)
        negative_dict_path : str
            Path to the file containing negative words (one per line)
        negation_window : int, default=3
            Number of words to look back from a sentiment word to check for negation
        intensifiers : Set[str], optional
            Set of intensifier words that amplify sentiment (e.g., 'very', 'extremely')
        """
        self.positive_dict_path = positive_dict_path
        self.negative_dict_path = negative_dict_path
        self.negation_window = negation_window
        
        # Default intensifiers
        if intensifiers is None:
            self.intensifiers = {
                'very', 'extremely', 'incredibly', 'absolutely', 'definitely',
                'truly', 'deeply', 'greatly', 'highly', 'really', 'quite',
                'so', 'much', 'far', 'far', 'way', 'awfully', 'terribly'
            }
        else:
            self.intensifiers = intensifiers
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
            'none', 'without', "n't", 'rarely', 'seldom', 'barely', 'hardly',
            'scarcely', 'isn', 'aren', 'wasn', 'weren', 'hasn', 'haven',
            'hadn', 'doesn', 'don', 'did', 'didn', 'won', 'wouldn',
            'shan', 'shouldn', 'can', 'couldn', 'mightn', 'mustn'
        }
        
        # Load dictionaries
        self._load_dictionaries()
        
        logger.info(f"Loaded {len(self.positive_words)} positive words")
        logger.info(f"Loaded {len(self.negative_words)} negative words")
    
    def _load_dictionaries(self):
        """Load positive and negative word dictionaries."""
        try:
            with open(self.positive_dict_path, 'r', encoding='utf-8') as f:
                self.positive_words = set(word.strip().lower() for word in f if word.strip())
        except FileNotFoundError:
            logger.warning(f"Positive dictionary not found at {self.positive_dict_path}")
            self.positive_words = set()
        
        try:
            with open(self.negative_dict_path, 'r', encoding='utf-8') as f:
                self.negative_words = set(word.strip().lower() for word in f if word.strip())
        except FileNotFoundError:
            logger.warning(f"Negative dictionary not found at {self.negative_dict_path}")
            self.negative_words = set()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text into tokens.
        
        Parameters:
        -----------
        text : str
            Input text to preprocess
            
        Returns:
        --------
        List[str]
            List of preprocessed tokens
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep words
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        return tokens
    
    def _is_negated(self, tokens: List[str], word_position: int) -> bool:
        """
        Check if a word is negated by looking back at previous tokens.
        
        Parameters:
        -----------
        tokens : List[str]
            List of tokens
        word_position : int
            Position of the word to check
            
        Returns:
        --------
        bool
            True if the word is negated, False otherwise
        """
        # Check words in the negation window before this word
        start_pos = max(0, word_position - self.negation_window)
        
        for i in range(start_pos, word_position):
            if tokens[i] in self.negation_words:
                return True
        
        return False
    
    def _has_intensifier(self, tokens: List[str], word_position: int) -> bool:
        """
        Check if a sentiment word is preceded by an intensifier.
        
        Parameters:
        -----------
        tokens : List[str]
            List of tokens
        word_position : int
            Position of the word to check
            
        Returns:
        --------
        bool
            True if preceded by intensifier, False otherwise
        """
        if word_position > 0 and tokens[word_position - 1] in self.intensifiers:
            return True
        return False
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a given text.
        
        This method performs comprehensive sentiment analysis including:
        - Counting positive and negative words
        - Handling negations that reverse polarity
        - Amplifying scores for intensifiers
        - Computing final sentiment score and label
        
        Parameters:
        -----------
        text : str
            Input text to analyze
            
        Returns:
        --------
        Dict
            Dictionary containing:
            - 'positive_count': Number of positive words
            - 'negative_count': Number of negative words
            - 'negated_positive': Positive words that were negated
            - 'negated_negative': Negative words that were negated
            - 'score': Final sentiment score (-1 to 1)
            - 'label': Sentiment label (positive, negative, neutral)
            - 'confidence': Confidence score (0 to 1)
        """
        tokens = self._preprocess_text(text)
        
        if not tokens:
            return {
                'positive_count': 0,
                'negative_count': 0,
                'negated_positive': 0,
                'negated_negative': 0,
                'score': 0.0,
                'label': 'neutral',
                'confidence': 0.0,
                'raw_pos_score': 0.0,
                'raw_neg_score': 0.0
            }
        
        positive_count = 0
        negative_count = 0
        negated_positive = 0
        negated_negative = 0
        positive_score = 0.0
        negative_score = 0.0
        
        # Scan through tokens
        for idx, token in enumerate(tokens):
            is_negated = self._is_negated(tokens, idx)
            has_intensifier = self._has_intensifier(tokens, idx)
            
            # Intensifier multiplier
            intensity = 1.5 if has_intensifier else 1.0
            
            # Check for positive words
            if token in self.positive_words:
                if is_negated:
                    negated_positive += 1
                    negative_score += 1.0 * intensity
                else:
                    positive_count += 1
                    positive_score += 1.0 * intensity
            
            # Check for negative words
            elif token in self.negative_words:
                if is_negated:
                    negated_negative += 1
                    positive_score += 1.0 * intensity
                else:
                    negative_count += 1
                    negative_score += 1.0 * intensity
        
        # Calculate final score (normalized)
        total_sentiment_words = positive_count + negative_count + negated_positive + negated_negative
        
        if total_sentiment_words == 0:
            return {
                'positive_count': 0,
                'negative_count': 0,
                'negated_positive': 0,
                'negated_negative': 0,
                'score': 0.0,
                'label': 'neutral',
                'confidence': 0.0,
                'raw_pos_score': 0.0,
                'raw_neg_score': 0.0
            }
        
        # Normalize scores
        final_positive_score = positive_score + negated_negative
        final_negative_score = negative_score + negated_positive
        
        # Calculate normalized score (-1 to 1)
        raw_score = final_positive_score - final_negative_score
        normalized_score = raw_score / (final_positive_score + final_negative_score) if (final_positive_score + final_negative_score) > 0 else 0
        normalized_score = max(-1.0, min(1.0, normalized_score))
        
        # Determine label and confidence
        if normalized_score > 0.1:
            label = 'positive'
            confidence = min(1.0, abs(normalized_score))
        elif normalized_score < -0.1:
            label = 'negative'
            confidence = min(1.0, abs(normalized_score))
        else:
            label = 'neutral'
            confidence = 1.0 - abs(normalized_score)
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'negated_positive': negated_positive,
            'negated_negative': negated_negative,
            'score': normalized_score,
            'label': label,
            'confidence': confidence,
            'raw_pos_score': final_positive_score,
            'raw_neg_score': final_negative_score
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for a batch of texts.
        
        Parameters:
        -----------
        texts : List[str]
            List of texts to analyze
            
        Returns:
        --------
        List[Dict]
            List of sentiment analysis results
        """
        return [self.analyze_sentiment(text) for text in texts]


def process_csv_file(input_path: str, output_path: str, text_column: str,
                     analyzer: BingLiuSentimentAnalyzer, 
                     sentiment_column_name: str = 'bing_liu_sentiment') -> pd.DataFrame:
    """
    Process a CSV file and add Bing Liu sentiment analysis.
    
    Parameters:
    -----------
    input_path : str
        Path to input CSV file
    output_path : str
        Path to output CSV file
    text_column : str
        Name of the column containing text to analyze
    analyzer : BingLiuSentimentAnalyzer
        Initialized sentiment analyzer
    sentiment_column_name : str
        Name for the new sentiment column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with sentiment analysis added
    """
    try:
        logger.info(f"Reading {input_path}...")
        df = pd.read_csv(input_path)
        
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in {input_path}")
            logger.error(f"Available columns: {list(df.columns)}")
            return df
        
        logger.info(f"Analyzing sentiment for {len(df)} rows...")
        
        # Analyze sentiment for each row
        sentiments = []
        for idx, text in enumerate(df[text_column]):
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} rows...")
            
            result = analyzer.analyze_sentiment(str(text))
            sentiments.append(result['label'])
        
        # Add sentiment column
        df[sentiment_column_name] = sentiments
        
        # Optionally add detailed scores
        results = analyzer.analyze_batch(df[text_column].astype(str).tolist())
        df[f'{sentiment_column_name}_score'] = [r['score'] for r in results]
        df[f'{sentiment_column_name}_confidence'] = [r['confidence'] for r in results]
        
        logger.info(f"Writing results to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info(f"Completed! Saved to {output_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print(f"SENTIMENT ANALYSIS SUMMARY: {input_path}")
        print("="*60)
        print(df[sentiment_column_name].value_counts())
        print(f"Average confidence: {df[f'{sentiment_column_name}_confidence'].mean():.4f}")
        print("="*60 + "\n")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        raise


def process_all_files(base_path: str, preprocessing_temp_path: str,
                      positive_dict_path: str, negative_dict_path: str):
    """
    Process all CSV files in preprocessing_temp directory.
    
    Parameters:
    -----------
    base_path : str
        Base path of the project
    preprocessing_temp_path : str
        Path to preprocessing_temp directory
    positive_dict_path : str
        Path to positive words dictionary
    negative_dict_path : str
        Path to negative words dictionary
    """
    # Initialize analyzer
    analyzer = BingLiuSentimentAnalyzer(
        positive_dict_path=positive_dict_path,
        negative_dict_path=negative_dict_path,
        negation_window=3
    )
    
    # Define files to process with their respective text columns
    files_to_process = [
        {
            'filename': 'Cleaned_Iran_War_Sentiment_style_original.csv',
            'text_column': 'final_text_original',
            'output_suffix': '_with_bing_liu'
        },
        {
            'filename': 'Cleaned_Iran_War_Sentiment_style_b.csv',
            'text_column': 'final_text_style_b',
            'output_suffix': '_with_bing_liu'
        },
        {
            'filename': 'Cleaned_Iran_War_Sentiment_style_c.csv',
            'text_column': 'final_text_style_c',
            'output_suffix': '_with_bing_liu'
        }
    ]
    
    # Process each file
    for file_config in files_to_process:
        input_file = os.path.join(preprocessing_temp_path, file_config['filename'])
        output_file = os.path.join(
            preprocessing_temp_path,
            file_config['filename'].replace('.csv', f"{file_config['output_suffix']}.csv")
        )
        
        if os.path.exists(input_file):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {file_config['filename']}")
            logger.info(f"Text column: {file_config['text_column']}")
            logger.info(f"{'='*60}")
            
            process_csv_file(
                input_path=input_file,
                output_path=output_file,
                text_column=file_config['text_column'],
                analyzer=analyzer,
                sentiment_column_name='bing_liu_sentiment'
            )
        else:
            logger.warning(f"File not found: {input_file}")


if __name__ == '__main__':
    # Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_PATH = os.path.dirname(SCRIPT_DIR)  # For process_all_files if needed
    PREPROCESSING_TEMP_PATH = os.path.join(BASE_PATH, 'preprocessing_temp')
    POSITIVE_DICT_PATH = os.path.join(SCRIPT_DIR, 'positive-words.txt')
    NEGATIVE_DICT_PATH = os.path.join(SCRIPT_DIR, 'negative-words.txt')
    
    # Verify paths exist
    if not os.path.exists(PREPROCESSING_TEMP_PATH):
        logger.error(f"Preprocessing temp path not found: {PREPROCESSING_TEMP_PATH}")
        exit(1)
    
    if not os.path.exists(POSITIVE_DICT_PATH):
        logger.error(f"Positive dictionary not found: {POSITIVE_DICT_PATH}")
        exit(1)
    
    if not os.path.exists(NEGATIVE_DICT_PATH):
        logger.error(f"Negative dictionary not found: {NEGATIVE_DICT_PATH}")
        exit(1)
    
    # Process all files
    logger.info("Starting Bing Liu Sentiment Analysis Pipeline")
    logger.info(f"Base path: {BASE_PATH}")
    logger.info(f"Processing temp path: {PREPROCESSING_TEMP_PATH}")
    
    process_all_files(
        base_path=BASE_PATH,
        preprocessing_temp_path=PREPROCESSING_TEMP_PATH,
        positive_dict_path=POSITIVE_DICT_PATH,
        negative_dict_path=NEGATIVE_DICT_PATH
    )
    
    logger.info("\nSentiment analysis pipeline completed successfully!")
