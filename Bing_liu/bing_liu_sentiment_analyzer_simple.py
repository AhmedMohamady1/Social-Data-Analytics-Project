"""
Simple Bing Liu Sentiment Analyzer
===================================
Analyzes text sentiment using positive and negative word lists.
Handles negation words (like 'not') that reverse sentiment.
Compares results with ground truth labels and calculates accuracy.
"""

import pandas as pd
import os
import re

# Print info and error messages
def print_info(message):
    print(f"[INFO] {message}")

def print_error(message):
    print(f"[ERROR] {message}")


def calculate_accuracy(df, sentiment_name):
    """
    Calculate accuracy by comparing bing_liu_sentiment with ground_truth.
    
    Returns dict with accuracy metrics for each sentiment class and overall.
    """
    if 'ground_truth' not in df.columns:
        return {}
    
    total = len(df)
    correct = (df[sentiment_name] == df['ground_truth']).sum()
    
    # Calculate per-class accuracy
    sentiments = ['positive', 'negative', 'neutral']
    accuracy_dict = {
        'overall': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total
    }
    
    for sentiment in sentiments:
        mask = df['ground_truth'] == sentiment
        if mask.sum() > 0:
            class_correct = ((df[sentiment_name] == df['ground_truth']) & mask).sum()
            accuracy_dict[sentiment] = class_correct / mask.sum()
        else:
            accuracy_dict[sentiment] = 0.0
    
    return accuracy_dict


def save_summary(df, sentiment_name, output_summary_path):
    """Save sentiment analysis summary to a file."""
    try:
        with open(output_summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"SENTIMENT ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            # Sentiment distribution
            f.write("Sentiment Distribution:\n")
            f.write("-" * 60 + "\n")
            sentiment_counts = df[sentiment_name].value_counts()
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"{sentiment}: {count} ({percentage:.2f}%)\n")
            
            f.write("\n")
            f.write("Sentiment Scores:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Average score: {df[f'{sentiment_name}_score'].mean():.4f}\n")
            f.write(f"Min score: {df[f'{sentiment_name}_score'].min():.4f}\n")
            f.write(f"Max score: {df[f'{sentiment_name}_score'].max():.4f}\n")
            f.write(f"Std dev: {df[f'{sentiment_name}_score'].std():.4f}\n")
            
            f.write("\n")
            f.write("Confidence Scores:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Average confidence: {df[f'{sentiment_name}_confidence'].mean():.4f}\n")
            f.write(f"Min confidence: {df[f'{sentiment_name}_confidence'].min():.4f}\n")
            f.write(f"Max confidence: {df[f'{sentiment_name}_confidence'].max():.4f}\n")
            
            # Add ground truth comparison if available
            if 'ground_truth' in df.columns:
                f.write("\n")
                f.write("Accuracy Comparison with Ground Truth:\n")
                f.write("-" * 60 + "\n")
                accuracy = calculate_accuracy(df, sentiment_name)
                f.write(f"Overall Accuracy: {accuracy['overall']:.4f} ({accuracy['correct']}/{accuracy['total']})\n")
                f.write(f"Positive Accuracy: {accuracy['positive']:.4f}\n")
                f.write(f"Negative Accuracy: {accuracy['negative']:.4f}\n")
                f.write(f"Neutral Accuracy: {accuracy['neutral']:.4f}\n")
            
            f.write("\n")
            f.write("="*60 + "\n")
        
        print_info(f"Summary saved to {output_summary_path}")
    except Exception as e:
        print_error(f"Error saving summary: {str(e)}")


class SimpleSentimentAnalyzer:
    """Analyzes sentiment using positive and negative word dictionaries."""
    
    def __init__(self, positive_file, negative_file, negation_window=3):
        """
        Load word lists and setup analyzer.
        
        positive_file: path to file with positive words (one per line)
        negative_file: path to file with negative words (one per line)
        negation_window: how many words back to check for negations
        """
        self.positive_file = positive_file
        self.negative_file = negative_file
        self.negation_window = negation_window
        
        # Words that make sentiment stronger (very, extremely, etc.)
        self.intensifiers = {
            'very', 'extremely', 'incredibly', 'absolutely', 'definitely',
            'truly', 'deeply', 'greatly', 'highly', 'really', 'quite',
            'so', 'much', 'way', 'awfully', 'terribly'
        }
        
        # Words that flip sentiment (not, no, never, etc.)
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
            'none', 'without', "n't", 'rarely', 'seldom', 'barely', 'hardly',
            'scarcely', 'isn', 'aren', 'wasn', 'weren', 'hasn', 'haven',
            'hadn', 'doesn', 'don', 'didn', 'won', 'wouldn',
            'shan', 'shouldn', 'can', 'couldn', 'mightn', 'mustn'
        }
        
        # Load word lists
        self._load_words()
        print_info(f"Loaded {len(self.positive_words)} positive words")
        print_info(f"Loaded {len(self.negative_words)} negative words")
    
    def _load_words(self):
        """Load positive and negative words from files."""
        self.positive_words = set()
        try:
            with open(self.positive_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.positive_words.add(word)
        except FileNotFoundError:
            print_error(f"File not found: {self.positive_file}")
        
        self.negative_words = set()
        try:
            with open(self.negative_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.negative_words.add(word)
        except FileNotFoundError:
            print_error(f"File not found: {self.negative_file}")
    
    def _clean_text(self, text):
        """Convert text to lowercase words."""
        if not isinstance(text, str):
            return []
        
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Split into words
        words = text.split()
        return words
    
    def _check_negation(self, words, position):
        """Check if word is negated by looking back."""
        start = max(0, position - self.negation_window)
        
        for i in range(start, position):
            if words[i] in self.negation_words:
                return True
        
        return False
    
    def _check_intensifier(self, words, position):
        """Check if word before this one is an intensifier."""
        if position > 0 and words[position - 1] in self.intensifiers:
            return True
        return False
    
    def analyze(self, text):
        """
        Analyze sentiment and return results.
        
        Returns:
        - positive_count: number of positive words
        - negative_count: number of negative words
        - negated_positive: positive words that were negated
        - negated_negative: negative words that were negated
        - score: sentiment score (-1 to 1)
        - label: 'positive', 'negative', or 'neutral'
        - confidence: confidence score (0 to 1)
        """
        words = self._clean_text(text)
        
        if not words:
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
        
        pos_count = 0
        neg_count = 0
        neg_pos = 0
        neg_neg = 0
        pos_score = 0.0
        neg_score = 0.0
        
        # Check each word
        for i, word in enumerate(words):
            is_negated = self._check_negation(words, i)
            has_intensifier = self._check_intensifier(words, i)
            
            # Boost score if intensifier (1.5x instead of 1.0x)
            boost = 1.5 if has_intensifier else 1.0
            
            # Check positive words
            if word in self.positive_words:
                if is_negated:
                    neg_pos += 1
                    neg_score += 1.0 * boost
                else:
                    pos_count += 1
                    pos_score += 1.0 * boost
            
            # Check negative words
            elif word in self.negative_words:
                if is_negated:
                    neg_neg += 1
                    pos_score += 1.0 * boost
                else:
                    neg_count += 1
                    neg_score += 1.0 * boost
        
        # Calculate final score
        total = pos_count + neg_count + neg_pos + neg_neg
        
        if total == 0:
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
        
        # Combine scores (negated negative words count as positive)
        final_pos = pos_score + neg_neg
        final_neg = neg_score + neg_pos
        
        # Normalize to -1 to 1
        if final_pos + final_neg > 0:
            norm_score = (final_pos - final_neg) / (final_pos + final_neg)
        else:
            norm_score = 0
        
        norm_score = max(-1.0, min(1.0, norm_score))
        
        # Determine sentiment label
        if norm_score > 0.1:
            label = 'positive'
            confidence = min(1.0, abs(norm_score))
        elif norm_score < -0.1:
            label = 'negative'
            confidence = min(1.0, abs(norm_score))
        else:
            label = 'neutral'
            confidence = 1.0 - abs(norm_score)
        
        return {
            'positive_count': pos_count,
            'negative_count': neg_count,
            'negated_positive': neg_pos,
            'negated_negative': neg_neg,
            'score': norm_score,
            'label': label,
            'confidence': confidence,
            'raw_pos_score': final_pos,
            'raw_neg_score': final_neg
        }
    
    def analyze_many(self, text_list):
        """Analyze multiple texts at once."""
        results = []
        for text in text_list:
            results.append(self.analyze(text))
        return results


def process_file(input_path, output_path, text_column, analyzer, sentiment_name='sentiment'):
    """
    Read CSV, analyze sentiment, save results.
    
    input_path: path to input CSV
    output_path: path to output CSV
    text_column: name of column with text
    analyzer: sentiment analyzer
    sentiment_name: name for new sentiment column
    """
    try:
        print_info(f"Reading {input_path}...")
        df = pd.read_csv(input_path)
        
        if text_column not in df.columns:
            print_error(f"Column '{text_column}' not found")
            print_error(f"Available: {list(df.columns)}")
            return df
        
        print_info(f"Analyzing {len(df)} rows...")
        
        # Analyze each row
        labels = []
        for idx, text in enumerate(df[text_column]):
            if idx % 100 == 0:
                print_info(f"Progress: {idx}/{len(df)}")
            
            result = analyzer.analyze(str(text))
            labels.append(result['label'])
        
        # Add sentiment column
        df[sentiment_name] = labels
        
        # Add detailed scores
        results = analyzer.analyze_many(df[text_column].astype(str).tolist())
        df[f'{sentiment_name}_score'] = [r['score'] for r in results]
        df[f'{sentiment_name}_confidence'] = [r['confidence'] for r in results]
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        print_info(f"Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        print_info("Done!")
        
        # Save summary to separate file
        summary_path = output_path.replace('.csv', '_summary.txt')
        save_summary(df, sentiment_name, summary_path)
        
        # Show summary and accuracy
        print("\n" + "="*60)
        print(f"RESULTS: {input_path}")
        print("="*60)
        print(df[sentiment_name].value_counts())
        avg_confidence = df[f'{sentiment_name}_confidence'].mean()
        print(f"Average confidence: {avg_confidence:.4f}")
        
        # Print accuracy if ground_truth exists
        if 'ground_truth' in df.columns:
            print("\n" + "-"*60)
            print("ACCURACY COMPARISON WITH GROUND TRUTH:")
            print("-"*60)
            accuracy = calculate_accuracy(df, sentiment_name)
            print(f"Overall Accuracy: {accuracy['overall']:.4f} ({accuracy['correct']}/{accuracy['total']})")
            print(f"Positive Accuracy: {accuracy['positive']:.4f}")
            print(f"Negative Accuracy: {accuracy['negative']:.4f}")
            print(f"Neutral Accuracy: {accuracy['neutral']:.4f}")
        
        print("="*60 + "\n")
        
        return df
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        raise


if __name__ == '__main__':
    # Setup paths
    BASE_PATH = '/home/el3omda/projects/personal/SDA'
    TEMP_PATH = os.path.join(BASE_PATH, 'Social-Data-Analytics-Project/Task_3/preprocessing_temp')
    OUTPUT_PATH = os.path.join(BASE_PATH, 'Social-Data-Analytics-Project/Task_3/Bing_liu')
    POS_FILE = os.path.join(BASE_PATH, 'Social-Data-Analytics-Project/Task_3/Bing_liu/positive-words.txt')
    NEG_FILE = os.path.join(BASE_PATH, 'Social-Data-Analytics-Project/Task_3/Bing_liu/negative-words.txt')
    
    # Check all paths exist
    if not os.path.exists(TEMP_PATH):
        print_error(f"Not found: {TEMP_PATH}")
        exit(1)
    
    if not os.path.exists(POS_FILE):
        print_error(f"Not found: {POS_FILE}")
        exit(1)
    
    if not os.path.exists(NEG_FILE):
        print_error(f"Not found: {NEG_FILE}")
        exit(1)
    
    # Create analyzer
    analyzer = SimpleSentimentAnalyzer(
        positive_file=POS_FILE,
        negative_file=NEG_FILE,
        negation_window=3
    )
    
    # Files to process
    files = [
        {
            'name': 'Cleaned_Iran_War_Sentiment_style_original.csv',
            'column': 'final_text_original',
            'output_name': 'Cleaned_Iran_War_Sentiment_style_original_with_bing_liu.csv'
        },
        {
            'name': 'Cleaned_Iran_War_Sentiment_style_b.csv',
            'column': 'final_text_style_b',
            'output_name': 'Cleaned_Iran_War_Sentiment_style_b_with_bing_liu.csv'
        },
        {
            'name': 'Cleaned_Iran_War_Sentiment_style_c.csv',
            'column': 'final_text_style_c',
            'output_name': 'Cleaned_Iran_War_Sentiment_style_c_with_bing_liu.csv'
        }
    ]
    
    # Run
    print_info("Starting sentiment analysis")
    print_info(f"Base path: {BASE_PATH}")
    print_info(f"Temp path: {TEMP_PATH}")
    print_info(f"Output path: {OUTPUT_PATH}")
    
    # Process each file
    for file_info in files:
        input_file = os.path.join(TEMP_PATH, file_info['name'])
        output_file = os.path.join(OUTPUT_PATH, file_info['output_name'])
        
        if os.path.exists(input_file):
            print("\n" + "="*60)
            print(f"Processing: {file_info['name']}")
            print(f"Text column: {file_info['column']}")
            print(f"Output: {file_info['output_name']}")
            print("="*60)
            
            process_file(
                input_path=input_file,
                output_path=output_file,
                text_column=file_info['column'],
                analyzer=analyzer,
                sentiment_name='bing_liu_sentiment'
            )
        else:
            print_error(f"File not found: {input_file}")
    
    print_info("Sentiment analysis complete!")
