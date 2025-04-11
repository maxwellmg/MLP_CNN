from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Load dataset
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Vectorize data
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data)

# Convert the TF-IDF vectors to 32-bit (save on memory)
X_train_tfidf_32bit = X_train_tfidf.astype(np.float32)

# Treat y_train_original as the target variable
y_train_original = newsgroups_train.target

# Split into training and testing sets (80%-20% divide)
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf_32bit, y_train_original, test_size=0.2, random_state=42)

# Divide the training data into 75% training and 25% validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Create MLP model
mlp_model = MLPClassifier(random_state=42)

# Train model
mlp_model.fit(X_train_split, y_train_split)

# Evaluate the model on validation set
y_val_pred = mlp_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy}")

# Validation Accuracy: 0.7145382235969951


# Automatically tune to optimize the MLP model

mlp_model_2 = GridSearchCV(MLPClassifier(early_stopping=True),
                    {'hidden_layer_sizes': [(512,), (256, 128)],
                    'alpha': [0.0001, 0.001],
                    'learning_rate_init': [0.001, 0.0005]},
                    return_train_score=True,
                    scoring='accuracy',
                    cv=3)

# Train data

mlp_model_2.fit(X_train_split, y_train_split)

# Evaluate on validation set
y_val_pred_2 = mlp_model_2.predict(X_val)
accuracy_2 = accuracy_score(y_val, y_val_pred_2)
print(f"Model 2 Validation Accuracy: {accuracy_2}")
# Validation Accuracy: 0.723376049491825

#Computation Heavy MLP Model
mlp_model_3 = GridSearchCV(MLPClassifier(early_stopping=True),
 {'hidden_layer_sizes':
  [(512,),
   (512, 256),
   (512, 256, 128)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.0005, 0.0001],
    'batch_size': [128, 256],
    'learning_rate': ['adaptive']},
                    return_train_score=True,
                    cv=3,
                    n_jobs=-1)


# Train model
mlp_model_3.fit(X_train_split, y_train_split)

# Evaluate on validation set
y_val_pred_3 = mlp_model_3.predict(X_val)
accuracy_3 = accuracy_score(y_val, y_val_pred_3)
print(f"Model 3 Validation Accuracy: {accuracy_3}")

# Get results from model 2 Grid Search
results = mlp_model_2.cv_results_

# Training and validation scores
train_scores = results['mean_train_score']
val_scores = results['mean_test_score']
params = results['params']

# Print results for all hyperparameter combinations
for i in range(len(params)):
    print(f"Hyperparameters: {params[i]}")
    print(f"  Mean Training Accuracy: {train_scores[i]:.4f}")
    print(f"  Mean Validation Accuracy: {val_scores[i]:.4f}\n")

# Get predictions from models
y_pred_1 = mlp_model.predict(X_test)
y_pred_2 = mlp_model_2.predict(X_test)

# Calculate metrics
metrics = {
    'Model 1': {
        'Accuracy': accuracy_score(y_test, y_pred_1),
        'Precision': precision_score(y_test, y_pred_1, average='weighted'),
        'Recall': recall_score(y_test, y_pred_1, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred_1, average='weighted')
    },
    'Model 2': {
        'Accuracy': accuracy_score(y_test, y_pred_2),
        'Precision': precision_score(y_test, y_pred_2, average='weighted'),
        'Recall': recall_score(y_test, y_pred_2, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred_2, average='weighted')
    }
}

# Create comparison table
comparison_df = pd.DataFrame(metrics).T.round(4)

print("Model Comparison Table:")
display(comparison_df)