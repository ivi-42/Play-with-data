import tensorflow as tf
import numpy as np
import requests
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
from alibi.explainers import AnchorText
from sklearn.inspection import permutation_importance
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Load spaCy model
nlp = spacy.load("en_core_web_sm")



# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Fetch data from Hacker News API
def fetch_hn_data(n_posts=10000):
    top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty"
    top_stories = requests.get(top_stories_url).json()[:n_posts]
    data = []
    for story_id in top_stories:
        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json?print=pretty"
        story = requests.get(story_url).json()
        score = story.get("score", 0)
        text = story.get("text", "") or story.get("title", "")
        data.append((text, score))
    return data

# Process text and extract features
def process_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Extract common words from compliant posts
def extract_common_words(data, threshold=50):
    all_words = []
    for text, score in data:
        if score > threshold:
            tokens = process_text(text)
            all_words.extend(tokens)
    word_freq = Counter(all_words)
    common_words = set(word for word, freq in word_freq.most_common(100))
    return common_words

# Convert text to feature vector based on common words
def text_to_features(text, common_words):
    tokens = set(process_text(text))
    features = [1 if word in tokens else 0 for word in common_words]
    return features


# Define the function to transform raw text into feature vectors
def text_to_feature_vector(text):
    processed_text = process_text(text)
    features = [1 if word in processed_text else 0 for word in common_words]
    return np.array(features, dtype=np.float32)

# Define the modified predictor function for AnchorText
def modified_predictor(texts):
    feature_vectors = np.vstack([text_to_feature_vector(text) for text in texts])
    return classifier.predict_proba(feature_vectors)

# Instantiate the AnchorText explainer
anchor_explainer = AnchorText(nlp=nlp, predictor=modified_predictor)




# Load data
hn_data = fetch_hn_data()
common_words = extract_common_words(hn_data)

# Feature extraction
X = np.array([text_to_features(text, common_words) for text, _ in hn_data], dtype=np.float32)
y = np.array([1 if score > 50 else 0 for _, score in hn_data], dtype=np.float32)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)



#after plot
# Get feature importances from the trained classifier
feature_importances = classifier.feature_importances_
# Feature names
feature_names = list(common_words)
##

# Visualize feature importances
importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
importances_df = importances_df.sort_values(by='importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=importances_df.head(20))  # Top 20 features
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# Permutation importance
perm_importance = permutation_importance(classifier, X_test, y_test)
sorted_perm_importances = sorted(zip(feature_names, perm_importance.importances_mean), key=lambda x: x[1], reverse=True)
for name, importance in sorted_perm_importances:
    print(f"{name}: {importance}")




# Define the modified predictor function for AnchorText
def modified_predictor(texts):
    feature_vectors = np.array([text_to_feature_vector(text) for text in texts])
    return classifier.predict_proba(feature_vectors)



# After training the classifier, show feature importances
importances = classifier.feature_importances_
feature_names = list(common_words)


#continue here OUT PLOT
sorted_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for name, importance in sorted_importances:
    print(f"{name}: {importance}")

# Perform permutation importance
perm_importance = permutation_importance(classifier, X_test, y_test)
sorted_perm_importances = sorted(zip(feature_names, perm_importance.importances_mean), key=lambda x: x[1], reverse=True)
for name, importance in sorted_perm_importances:
    print(f"{name}: {importance}")





# After training the classifier, show feature importances
#importances = classifier.feature_importances_
#feature_names = list(common_words)
#sorted_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
#for name, importance in sorted_importances:
#    print(f"{name}: {importance}")'



# Perform permutation importance
perm_importance = permutation_importance(classifier, X_test, y_test)
sorted_perm_importances = sorted(zip(common_words, perm_importance.importances_mean), key=lambda x: x[1], reverse=True)
for name, importance in sorted_perm_importances:
    print(f"{name}: {importance}")




# SHAP Analysis - Bypassing additivity check or using interventional perturbation
explainer = shap.TreeExplainer(classifier, check_additivity=False)
shap_values = explainer.shap_values(X_test, check_additivity=False)

shap.initjs()
class_to_visualize = 1
for i, instance in enumerate(X_test):
    shap_values_instance = explainer.shap_values(instance.reshape(1, -1), check_additivity=False)
    shap.force_plot(explainer.expected_value[class_to_visualize], shap_values_instance[class_to_visualize][0], instance)

# Define the modified predictor function for AnchorText
def modified_predictor(texts):
    feature_vectors = np.array([text_to_feature_vector(text) for text in texts])
    return classifier.predict_proba(feature_vectors)


# Instantiate the AnchorText explainer
anchor_explainer = AnchorText(nlp=nlp, predictor=modified_predictor)





# Evaluation and display
for i, instance in enumerate(X_test):
    prediction = classifier.predict(instance.reshape(1, -1))
    compliance = "compliant" if prediction[0] == 1 else "not compliant"
    original_text = hn_data[i][0]
    print(f"Post: {original_text}\nCompliance: {compliance}\n")

    # SHAP Analysis for each instance
    shap_values_instance = explainer.shap_values(instance.reshape(1, -1), check_additivity=False)

    shap.force_plot(explainer.expected_value[class_to_visualize], shap_values_instance[class_to_visualize][0], instance)

    # Anchor explanation for selected instances
    if i < 5:  # Adjust this number as needed
        text_instance = hn_data[i][0]
        print("Explaining:", text_instance)  # Debugging print
        explanation = anchor_explainer.explain(text_instance, threshold=0.95)
        print("Anchor:", ' AND '.join(explanation.anchor))
        print("Full explanation object:", explanation)  # Debugging print


