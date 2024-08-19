from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from datasets import load_dataset
import torch
import numpy as np

# Load the dataset
dataset = load_dataset('tweet_eval', 'sentiment')

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Extract TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_train = tfidf_vectorizer.fit_transform(dataset['train']['text']).toarray()
tfidf_val = tfidf_vectorizer.transform(dataset['validation']['text']).toarray()

# Extract BERT embeddings
def extract_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model.bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

bert_train_embeddings = np.array([extract_bert_embeddings(text) for text in dataset['train']['text']])
bert_val_embeddings = np.array([extract_bert_embeddings(text) for text in dataset['validation']['text']])

# Combine TF-IDF and BERT embeddings
X_train = np.hstack((tfidf_train, bert_train_embeddings))
X_val = np.hstack((tfidf_val, bert_val_embeddings))
y_train = np.array(dataset['train']['label'])
y_val = np.array(dataset['validation']['label'])

# Train a Random Forest model as part of the ensemble
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Create an ensemble model with majority voting
ensemble_model = VotingClassifier(estimators=[('bert', bert_model), ('rf', rf_model)], voting='soft')
ensemble_model.fit(X_train, y_train)

# Evaluate the model
accuracy = ensemble_model.score(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


#####

from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Ensure that the BERT model and other models are in evaluation mode
bert_model.eval()
rf_model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    
    # Get BERT embeddings
    bert_embedding = extract_bert_embeddings(data)
    
    # Get TF-IDF features
    tfidf_features = tfidf_vectorizer.transform([data]).toarray()
    
    # Combine features
    combined_features = np.hstack((tfidf_features, bert_embedding))
    
    # Predict with the ensemble model
    prediction = ensemble_model.predict(combined_features)
    sentiment = prediction[0]  # 0: Negative, 1: Neutral, 2: Positive
    
    return jsonify({'sentiment': int(sentiment)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
