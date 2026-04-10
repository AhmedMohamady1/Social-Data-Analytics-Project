import argparse
import pandas as pd
import re
import string
import emoji
import nltk
import os
import symspellpy
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from textblob import Word
from symspellpy import SymSpell
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
import time

# Ensure reproducible language detection
DetectorFactory.seed = 0

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

STOPWORDS = set(stopwords.words('english'))

class PreprocessingPipeline:
    def __init__(self):
        # Initialize SymSpell for spell correction
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        sym_path = os.path.dirname(symspellpy.__file__)
        dictionary_path = os.path.join(sym_path, "frequency_dictionary_en_82_765.txt")
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        
        # Initialize Lemmatizer once here to avoid slowing down the pandas .apply() loop
        self.lemmatizer = WordNetLemmatizer()

    def handle_language(self, df, column, mode='drop'):
        """Handles non-English rows: either dropping them or translating them"""
        def detect_lang(text):
            try: return detect(str(text))
            except: return 'unknown'

        print(f"Detecting languages and applying '{mode}' strategy...")
        df['lang'] = df[column].apply(detect_lang)

        if mode == 'drop':
            df = df[df['lang'] == 'en'].reset_index(drop=True)
        elif mode == 'translate':
            # tqdm for translation progress visibility
            tqdm.pandas(desc="Translating to English")
            df[column] = df.progress_apply(
                lambda x: x[column] if x['lang'] == 'en' else self.translate_text(x[column]),
                axis=1
            )
        
        return df.drop(columns=['lang'])

    def translate_text(self, text):
        """Translates text to English using deep_translator"""
        try:
            # GoogleTranslator is synchronous and doesn't require 'await'
            return GoogleTranslator(source='auto', target='en').translate(str(text))
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def convert_emojis(self, text):
        """Transforms visual icons into text descriptions."""
        return emoji.demojize(str(text))

    # Modular cleaning functions
    def remove_mastodon_artifacts(self, text):
        return re.sub(r'&quot;|quot|RE:', '', str(text), flags=re.IGNORECASE)
    def remove_urls(self, text): 
        return re.sub(r'https?://\S+|www\.\S+', '', str(text))
    def remove_html_tags(self, text):
        return re.sub(r'<.*?>', '', str(text))
    def remove_social_tags(self, text):
        return re.sub(r'@\w+|#\w+', '', str(text))
    def remove_numbers(self, text):
        return re.sub(r'\d+', '', str(text))
    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', str(text))
    def normalize_whitespace(self, text):
        return re.sub(r'\s+', ' ', str(text)).strip()
    def remove_stopwords(self, text):
        return " ".join([w for w in str(text).split() if w.lower() not in STOPWORDS])
    def fix_spelling(self, text):
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0].term
        
    def lemmatize_text(self, text):
        """Lemmatizes text using NLTK POS tagging to accurately handle verbs, nouns, and adjectives."""

        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN 

        tokens = word_tokenize(str(text))
        pos_tokens = pos_tag(tokens)
        
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
            for word, tag in pos_tokens
        ]
        return " ".join(cleaned_tokens)

    def extract_subject_tags(self, text):
        """Regex Extraction for partitioning tags."""
        text = text.lower()
        if any(w in text for w in ['protest', 'embassy', 'rally']): return "Protest/Activism"
        if any(w in text for w in ['drone', 'missile', 'attack']): return "Military Action"
        return "General Conflict"

    def predict_stance(self, df, api_key):
        """Uses gemini-3.1-flash-lite-preview with built-in rate limiting (15 RPM)."""
        
        # Inner helper function to clean up Gemini's occasional JSON formatting
        def clean_stance(text):
            text = str(text)
            match = re.search(r'(Anti-Iran|Pro-Iran|Neutral)', text, re.IGNORECASE)
            if match:
                # Standardizes capitalization regardless of how Gemini outputted it
                label = match.group(1).lower()
                if label == 'anti-iran': return 'Anti-Iran'
                if label == 'pro-iran': return 'Pro-Iran'
                return 'Neutral'
            return "Unknown"

        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", google_api_key=api_key)
        template = "Analyze the post stance: 'Pro-Iran', 'Anti-Iran', or 'Neutral'. Return only the label.\n\nPost: {post}"
        prompt = PromptTemplate(template=template, input_variables=["post"])
        chain = prompt | llm

        results = []
        print(f"Requesting Stance Analysis (Rate Limited to 15 RPM)...")
        
        for text in tqdm(df['sentiment_text'], desc="AI Processing"):
            try:
                response = chain.invoke({"post": str(text)[:500]})
                
                # Robust content extraction
                if hasattr(response, 'content'):
                    res_text = str(response.content).strip()
                else:
                    res_text = str(response).strip()
                
                # Apply the regex cleaner directly to the response before appending
                cleaned_res = clean_stance(res_text)
                results.append(cleaned_res)
                
                # Wait 4.1 seconds before the next loop iteration
                time.sleep(4.1) 
                
            except Exception as e:
                # If we still hit a rate limit, wait longer (Back-off)
                if "429" in str(e):
                    print("\nRate limit hit! Sleeping for 60 seconds...")
                    time.sleep(60)
                    results.append("Retry_Needed")
                else:
                    print(f"\nError: {e}")
                    results.append("Unknown")
        
        df['stance_category'] = results
        return df

def main():
    parser = argparse.ArgumentParser(description="Configurable Preprocessing Pipeline")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, help="Limit rows processed BEFORE translation")
    parser.add_argument("--lang_mode", type=str, choices=['drop', 'translate'], default='drop')
    parser.add_argument("--convert_emojis", action="store_true")
    parser.add_argument("--remove_mastodon_artifacts", action="store_true")
    parser.add_argument("--remove_urls", action="store_true")
    parser.add_argument("--remove_html_tags", action="store_true")
    parser.add_argument("--remove_social_tags", action="store_true")
    parser.add_argument("--remove_numbers", action="store_true")
    parser.add_argument("--remove_punctuation", action="store_true")
    parser.add_argument("--normalize_whitespace", action="store_true")
    parser.add_argument("--remove_stopwords", action="store_true")
    parser.add_argument("--fix_spelling", action="store_true")
    parser.add_argument("--lemmatize", action="store_true")
    parser.add_argument("--extract_tags", action="store_true")
    parser.add_argument("--gemini_stance", type=str)

    args = parser.parse_args()
    pipeline = PreprocessingPipeline()
    df = pd.read_csv(args.input)

    # 1. Handle Language
    df = pipeline.handle_language(df, 'sentiment_text', mode=args.lang_mode)

    # 2. Apply Limit
    if args.limit and args.limit < len(df):
        print(f"Limiting dataset to the first {args.limit} rows of valid data...")
        df = df.head(args.limit).copy()

    # 3. Modular Processing
    processed = df['sentiment_text'].copy()
    if args.convert_emojis:
        processed = processed.apply(pipeline.convert_emojis)
    if args.remove_mastodon_artifacts:
        processed = processed.apply(pipeline.remove_mastodon_artifacts)
    if args.remove_urls:
        processed = processed.apply(pipeline.remove_urls)
    if args.remove_html_tags:
        processed = processed.apply(pipeline.remove_html_tags)
    if args.remove_social_tags:
        processed = processed.apply(pipeline.remove_social_tags)
    if args.remove_numbers:
        processed = processed.apply(pipeline.remove_numbers)
    if args.remove_punctuation:
        processed = processed.apply(pipeline.remove_punctuation)
    if args.normalize_whitespace:
        processed = processed.apply(pipeline.normalize_whitespace)

    processed = processed.apply(lambda x: str(x).lower())

    if args.fix_spelling:
        processed = processed.apply(pipeline.fix_spelling)
    if args.remove_stopwords:
        processed = processed.apply(pipeline.remove_stopwords)
    if args.lemmatize:
        processed = processed.apply(pipeline.lemmatize_text)

    # 4. Categorization Tag
    if args.extract_tags:
        df['subject_tag'] = df['sentiment_text'].apply(pipeline.extract_subject_tags)
    
    if args.gemini_stance:
        df = pipeline.predict_stance(df, args.gemini_stance)

    df['final_text'] = processed
    df.to_csv(args.output, index=False)
    print(f"\nSuccess! Refined dataset saved to: {args.output}")

if __name__ == "__main__":
    main()