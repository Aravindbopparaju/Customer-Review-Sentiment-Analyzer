# Customer Review Sent Analyzer

## 📌 Project Description

Customer Review Sent Analyzer is an end-to-end Machine Learning project that classifies customer reviews into:

- Positive
- Neutral
- Negative

The system uses Natural Language Processing (NLP) techniques and Logistic Regression to analyze textual feedback and predict sentiment along with probability-based confidence scores.

This project demonstrates the complete Machine Learning workflow from data preprocessing to model development and application interface creation.

---

## 🎯 Project Objectives

- Perform sentiment classification on customer reviews
- Apply NLP preprocessing techniques
- Convert text data into numerical features using TF-IDF
- Train and evaluate a supervised classification model
- Build an interactive web interface using Streamlit

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Preprocessing
- Extracted numerical ratings from raw text
- Converted ratings into sentiment labels
- Removed special characters and noise
- Converted text to lowercase
- Handled missing values

### 2️⃣ Feature Engineering
- TF-IDF Vectorization
- Unigrams and Bigrams
- Stopword removal
- Maximum 10,000 features

### 3️⃣ Model Training
- Logistic Regression
- Class balancing for imbalanced dataset
- Stratified train-test split
- Evaluation using Accuracy, Precision, Recall, and F1-score

### 4️⃣ Model Performance
- Accuracy: ~86%
- Balanced handling of Positive, Neutral, and Negative classes

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Git & GitHub

---

## 📂 Project Structure

Customer review analyzer
│
├── data/
│ └── Amazon_Reviews.csv
│
├── model/
│ ├── sentiment_model.pkl
│ └── vectorizer.pkl
│
├── notebooks/
│ └── model_training.ipynb
│
├── model_training.py
├── app.py
└── requirements.txt

## 📊 Example Predictions

| Review | Predicted Sentiment |
|--------|--------------------|
| The product quality is excellent | Positive |
| It is okay, nothing special | Neutral |
| Worst product ever | Negative |

---

## 📚 Key Learning Outcomes

- Understanding NLP preprocessing techniques
- Handling imbalanced datasets
- Applying TF-IDF feature extraction
- Building classification models using Logistic Regression
- Creating interactive ML dashboards
- Structuring end-to-end ML projects
