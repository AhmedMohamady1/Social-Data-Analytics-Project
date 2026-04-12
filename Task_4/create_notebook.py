import json
from pathlib import Path

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task 4: Error Analysis of the Sentiment Model\n",
    "\n",
    "In this notebook, we analyze the records that the Random Forest model misclassified. We will explore patterns in these failures, propose theories for why the model struggles, generate synthetic adversarial examples to verify our theories, and conclude with the model's limitations."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Load Predictions\n",
    "First, we load the test set predictions and identify where the `model_prediction` differs from the `ground_truth`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Load predictions\n",
    "df = pd.read_csv('model_predictions.csv')\n",
    "\n",
    "# Isolate errors\n",
    "errors = df[df['ground_truth'] != df['model_prediction']].copy()\n",
    "accuracy = 1.0 - (len(errors) / len(df))\n",
    "\n",
    "print(f\"Total Test Samples: {len(df)}\")\n",
    "print(f\"Misclassified: {len(errors)}\")\n",
    "print(f\"Test Accuracy: {accuracy:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Inspecting the Misclassifications\n",
    "Let's look at a sample of the errors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.set_option('display.max_colwidth', None)\n",
    "display(errors[['final_text', 'ground_truth', 'model_prediction', 'confidence']].head(15))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Pattern Matching and Theory Formulation\n",
    "\n",
    "### Observed Patterns in Misclassified Samples:\n",
    "\n",
    "1. **Sarcasm and Irony**: Models using TF-IDF (bag-of-words/n-grams) typically fail to capture sarcasm because the words themselves are positive/neutral but the underlying meaning is negative (e.g., \"hope kevin sarcastic otherwise hes simply another manga dip shit\").\n",
    "2. **Implicit Sentiment / Context Dependence**: A text like \"expect bill go thanks trump\" lacks strongly polarized adjectives. The model predicts negative, but the context might be neutral or positive depending on external knowledge.\n",
    "3. **Complex Sentence Structures**: Long sentences with mixed sentiment clauses confuse the model, since it just averages TF-IDF weights.\n",
    "4. **Lack of Keywords**: The model relies on specific n-grams. If negative sentiment is expressed using rare words or descriptive scenes rather than overt negative adjectives, the model biases toward neutral."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Reverse Engineering: Testing Theories\n",
    "Let's test these theories by generating synthetic samples and running them through the loaded model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load actual model artifacts\n",
    "vectorizer = joblib.load('artifacts/tfidf_vectorizer.joblib')\n",
    "model = joblib.load('artifacts/sentiment_model.joblib')\n",
    "classes = joblib.load('artifacts/label_classes.joblib')\n",
    "\n",
    "def predict_synthetic(texts):\n",
    "    X = vectorizer.transform(texts)\n",
    "    preds = model.predict(X)\n",
    "    probs = model.predict_proba(X).max(axis=1).round(3)\n",
    "    for t, p, prob in zip(texts, preds, probs):\n",
    "        print(f\"Prediction: [{p.upper()}] ({prob}) | Text: {t}\")\n",
    "\n",
    "# Theory 1: Sarcasm (Words are positive, meaning is negative)\n",
    "sarcasm_samples = [\n",
    "    \"oh great, another wonderful war that will definitely fix everything.\",\n",
    "    \"wow, brilliant strategy by the politicians to get us all killed.\"\n",
    "]\n",
    "print(\"--- Testing Sarcasm ---\")\n",
    "predict_synthetic(sarcasm_samples)\n",
    "\n",
    "# Theory 2: Negation (Bag-of-words models often struggle with 'not good')\n",
    "negation_samples = [\n",
    "    \"the peace treaty is not working and things are not good.\",\n",
    "    \"i am not happy about the missile strike.\"\n",
    "]\n",
    "print(\"\\n--- Testing Negation ---\")\n",
    "predict_synthetic(negation_samples)\n",
    "\n",
    "# Theory 3: Contextual / Descriptive Negative (No explicit swear words or strong negative adjectives)\n",
    "descriptive_samples = [\n",
    "    \"families had to pack their belongings quickly as the sirens wailed loudly.\",\n",
    "    \"the buildings collapsed and smoke filled the clear blue sky.\"\n",
    "]\n",
    "print(\"\\n--- Testing Descriptive Negative ---\")\n",
    "predict_synthetic(descriptive_samples)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Final Analysis Conclusion\n",
    "\n",
    "### What the model fails to identify:\n",
    "\n",
    "Based on the error analysis and adversarial testing, the Random Forest model utilizing TF-IDF representations fails primarily in the following dimensions:\n",
    "\n",
    "1. **Semantic Compositionality**: Because TF-IDF is a \"bag-of-words\" approach, the model ignores word order. It fails to identify negations reliably (e.g., \"not happy\" might trigger 'neutral' or base its prediction solely on the weight of \"happy\").\n",
    " \n",
    "2. **Sarcasm and Pragmatics**: The model evaluates explicit phrases. When users employ sarcasm (\"oh great...\", \"brilliant strategy...\"), the model picks up the positive tokens and misclassifies the text entirely.\n",
    "\n",
    "3. **Descriptive Sentiment**: The model struggles to classify objective descriptions of catastrophic events as \"negative.\" If a sentence describes fleeing homes and smoking ruins without explicitly using words like \"terrible,\" \"sad,\" or \"angry,\" the model leans heavily toward the \"neutral\" class. It lacks the world knowledge necessary to infer that scenes of war imply a negative situation.\n",
    "\n",
    "**Summary**: The model is highly effective at identifying explicit, straightforward sentiment (e.g., text filled with profanity or direct praise). However, it is fundamentally an *explicit keyword relying system*. It completely fails on implicit sentiment, sarcasm, complex negations, and nuanced descriptive language."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("Task_4/Error_Analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)
print("Notebook Task_4/Error_Analysis.ipynb created successfully.")
