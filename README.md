#  Multi-class Classification of Tweets
## Project Overview
This project implements multi-class classification to categorize tweets into various classes based on their content. The tweets are classified into categories like sports & gaming, pop culture, business & entrepreneurs, daily life, science & technology, and arts & culture. The goal of this project is to predict the class of a tweet using machine learning models like Random Forest, Decision Tree, Support Vector Machine, and Logistic Regression.
- Table of Contents
- Introduction
- Installation
- Usage
- Data
- Preprocessing
- Models
- Evaluation
- Libraries
## Introduction
The aim of this project is to classify tweets into one of several categories using text processing and various machine learning algorithms. The project includes data preprocessing steps such as removing stopwords, stemming, and lemmatization, and uses n-grams (uni-gram, bi-gram, tri-gram) for feature extraction.
## Installation
To set up the project, clone the repository and install the required libraries. Ensure you have Python 3.6 or above installed.
## Usage
- Place your dataset (tweets_dataset.csv) in the project directory.
- Run the tweet_classification.py script to execute the analysis.
- The cleaned and processed data will be saved in the project directory for further analysis.
## Data
The project uses a dataset containing tweets and their corresponding categories. The key columns in the dataset are:
- text: The tweet content.
- label: Numerical class label.
- label_name: The name of the category (e.g., sports & gaming, pop culture, etc.).
Example of categories:
- sports_&_gaming
- pop_culture
- business_&_entrepreneurs
- daily_life
- science_&_technology
- arts_&_culture
The dataset undergoes the following preprocessing steps:
- Checking for missing values.
- Removing special characters and punctuations.
- Converting all text to lowercase.
- Removing stopwords.
- Stemming and lemmatization to normalize text.
## Preprocessing
The following data preprocessing techniques are implemented:
- Stopword removal: Common English stopwords are removed using NLTK.
- Punctuation removal: Special characters and punctuations are removed.
- Stemming & Lemmatization: Words are normalized to their root forms using Porter Stemmer and WordNet Lemmatizer.
- TF-IDF Vectorization: Text is transformed into numerical features using n-grams.
## Models
The following machine learning models are implemented to classify the tweets:
- Random Forest Classifier: An ensemble of decision trees.
- Decision Tree Classifier: A tree-based model that splits data into branches based on feature conditions.
- Support Vector Machine (SVM): A classifier that works by finding a hyperplane that best separates the data into classes.
- Logistic Regression: A statistical model that uses a logistic function to model multi-class classification.
Each model is evaluated using accuracy scores and classification reports.
## Evaluation
The performance of each model is evaluated based on:
- Accuracy: The proportion of correctly predicted instances.
- Precision, Recall, F1-Score: Evaluated for each class to provide a detailed view of the model performance.
## Sample Results:
- Random Forest: 0.74 Accuracy (Unigram)
- Decision Tree: 0.65 Accuracy (Unigram)
- SVM: 0.81 Accuracy (Unigram)
- Logistic Regression: 0.77 Accuracy (Unigram)
## Evaluation for N-Grams:
Performance is also evaluated using n-grams (bi-gram, tri-gram, n-gram). Models show varying levels of accuracy based on the n-gram method used.
## Libraries
The project depends on the following Python libraries:
- pandas
- numpy
- seaborn
- matplotlib
- sklearn
- nltk
