import nltk
import pandas as pd
import os
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# Download resources
# ==============================
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# ==============================
# POS tag conversion
# ==============================
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return None

# ==============================
# Sentiment Function
# ==============================
negation_words = ["not", "no", "never"]

def sentiwordnet_sentiment(text):
    if not isinstance(text, str):
        return "Neutral"

    text = text.lower()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    pos_score = 0
    neg_score = 0
    count = 0
    negate_window = 0

    for word, tag in tagged:

        if len(word) < 3:
            continue

        if word in negation_words:
            negate_window = 3
            continue

        wn_tag = get_wordnet_pos(tag)
        if wn_tag is None:
            continue

        synsets = wordnet.synsets(word, pos=wn_tag)
        if not synsets:
            continue

        pos_avg = 0
        neg_avg = 0

        for syn in synsets:
            try:
                swn_syn = swn.senti_synset(syn.name())
                pos_avg += swn_syn.pos_score()
                neg_avg += swn_syn.neg_score()
            except:
                continue

        pos_avg /= len(synsets)
        neg_avg /= len(synsets)

        if negate_window > 0:
            pos_avg, neg_avg = neg_avg, pos_avg
            negate_window -= 1

        pos_score += pos_avg
        neg_score += neg_avg
        count += 1

    if count == 0:
        return "Neutral"

    final_score = (pos_score - neg_score) / count

    if final_score > 0.05:
        return "Positive"
    elif final_score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# ==============================
# Apply on ALL datasets
# ==============================
folder_path = "../preprocessing_temp"

for file_name in os.listdir(folder_path):

    if not file_name.endswith(".csv") or "working" in file_name:
        continue

    print("\n==============================")
    print(f"Processing: {file_name}")

    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    print("Columns:", df.columns.tolist())

    # 🔥 FIX: auto-detect correct text column
    text_col = None
    for col in df.columns:
        if "final_text" in col:
            text_col = col
            break

    if text_col is None:
        print("No valid text column found, skipping...")
        continue

    print(f"Using text column: {text_col}")

    # Clean data
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)

    # Apply model
    df["swn_sentiment"] = df[text_col].apply(sentiwordnet_sentiment)

    # Save results (ONE FILE PER DATASET)
    output_name = f"swn_{file_name}"
    df.to_csv(output_name, index=False)
    print(f"Saved: {output_name}")

    # ==============================
    # Evaluation
    # ==============================
    if "ground_truth" not in df.columns:
        print("No ground_truth column, skipping evaluation...")
        continue

    df["ground_truth"] = df["ground_truth"].astype(str).str.capitalize()
    df["swn_sentiment"] = df["swn_sentiment"].astype(str).str.capitalize()

    valid_labels = ["Positive", "Negative", "Neutral"]
    df = df[df["ground_truth"].isin(valid_labels)]
    df = df[df["swn_sentiment"].isin(valid_labels)]

    y_true = df["ground_truth"]
    y_pred = df["swn_sentiment"]

    accuracy = accuracy_score(y_true, y_pred)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

print("\n All datasets processed successfully!")