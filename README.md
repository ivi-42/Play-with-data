# Play-with-data

# Hacker News Compliance Classifier

## Overview
This script analyzes and classifies posts from Hacker News (HN) into 'compliant' and 'non-compliant' categories. It includes data fetching, preprocessing, feature extraction, machine learning model training, and model interpretation.

This is still a heavy workaround with many parts that are merely a generalization of a complex topic as human behaviour (https://www.youtube.com/watch?v=p0mRIhK9seg)

Use the Google colab to start it in cloud and see the script at work.

There are also 2 pdf in this repository, one called ai_basic_fast.pdf and one gianni.pojani_ai_basics.pdf. The first, is a digestible overview of the algorithms used, done by chatgpt, the second is a more verbose writing over the research done and the direction to take, with particular focus on experimentation and functionality.
## Key Components

### Data Fetching and Preprocessing
- **Function**: `fetch_hn_data(n_posts)`
- **Purpose**: Retrieves top stories from the Hacker News API and preprocesses the text.

### Feature Extraction
- **Functions**: 
  - `extract_common_words(data, threshold)`
  - `text_to_features(text, common_words)`
- **Purpose**: Identifies common words in high-scoring posts and converts posts into feature vectors.

### Machine Learning Model
- **Model**: `RandomForestClassifier`
- **Purpose**: Classifies posts into compliant and non-compliant categories.

### Model Interpretation and Analysis
- **Tools**: 
  - SHAP
  - AnchorText (Alibi)
  - Permutation Importance
- **Purpose**: Provide insights into the model’s decision-making process.

### Visualization and Output
- **Libraries**: 
  - Matplotlib
  - Seaborn
- **Purpose**: Visualize feature importance and display compliance status of posts.

## Script Breakdown

### Data Collection and Preparation
- Fetching and preprocessing data from Hacker News.
- Extracting features from the text data.

### Machine Learning Workflow
- Training the RandomForestClassifier.
- Applying SHAP and AnchorText for model interpretation.

### Outputs
- Visualizations of feature importances.
- Compliance status of HN posts.
- Anchor explanations for selected instances.

## Usage and Functionality

### Analysis and Interpretation
- Analyzing factors influencing post classification.
- Model interpretability for decision transparency.

### Applications
- Insights into content quality and user interactions.
- Ability to link semantic rules to their "manifestation" in online communities, better understanding of social dynamics

### Output Details
- **Compliance Status**: Whether a post is compliant according to the model.
- **SHAP Values**: Impact of each feature on the model's output.
- **Anchor Text Explanation**: Influential parts of text in model decision for a specific instance.



