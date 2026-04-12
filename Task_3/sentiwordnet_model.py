import nltk
import pandas as pd
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag


#nltk.download('sentiwordnet')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('averaged_perceptron_tagger_eng')



# POS tag conversion
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



# Sentiment Function
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

    negate_window = 0  # affects next 3 words

    for word, tag in tagged:

        # Skip short/noisy words
        if len(word) < 3:
            continue

        # Handle negation
        if word in negation_words:
            negate_window = 3
            continue

        wn_tag = get_wordnet_pos(tag)
        if wn_tag is None:
            continue

        synsets = wordnet.synsets(word, pos=wn_tag)
        if not synsets:
            continue

        # Average scores over all synsets
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

        # Apply negation
        if negate_window > 0:
            pos_avg, neg_avg = neg_avg, pos_avg
            negate_window -= 1

        pos_score += pos_avg
        neg_score += neg_avg
        count += 1

    if count == 0:
        return "Neutral"

    # Normalize score
    final_score = (pos_score - neg_score) / count

    # Threshold for Neutral
    if final_score > 0.05:
        return "Positive"
    elif final_score < -0.05:
        return "Negative"
    else:
        return "Neutral"



# Apply on Dataset
df = pd.read_csv("Task_3/Cleaned_Iran_War_Sentiment_with_Sentiment_Labels.csv")

# Clean data
df = df.dropna(subset=["final_text"])
df["final_text"] = df["final_text"].astype(str)

# Apply model
df["swn_sentiment"] = df["final_text"].apply(sentiwordnet_sentiment)

# Save results
df.to_csv("Task_3/swn_output.csv", index=False)

print("Sentiment prediction completed!")




# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Normalize labels
df["ground_truth"] = df["ground_truth"].astype(str).str.capitalize()
df["swn_sentiment"] = df["swn_sentiment"].astype(str).str.capitalize()

valid_labels = ["Positive", "Negative", "Neutral"]
df = df[df["ground_truth"].isin(valid_labels)]
df = df[df["swn_sentiment"].isin(valid_labels)]

y_true = df["ground_truth"]
y_pred = df["swn_sentiment"]

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=3))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))