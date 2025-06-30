# 📱 SMS Spam Text Classification

A deep learning-based binary text classifier that identifies whether an incoming SMS message is **spam** or **ham (not spam)**. This project leverages TensorFlow and NLP techniques to create a robust and scalable message filtering system.

## 🔍 Problem Statement

SMS spam is a major issue, often containing malicious content or wasting user time. This project aims to build a model that classifies SMS messages as spam or ham using supervised learning techniques on a real-world dataset.

## 🚀 Project Overview

- Loaded and explored a labeled SMS dataset.
- Preprocessed the text data using TensorFlow's TextVectorization layer.
- Built and trained a deep learning model using Keras Sequential API.
- Evaluated the model using standard metrics and visualized training performance.

## 📂 Dataset

- Format: `.tsv` (Tab-separated values)
- Total messages: ~5,500
- Labels: `spam` and `ham`
- Source: [FreeCodeCamp GitHub - SMS Dataset](https://github.com/beaucarnes/fcc_python_curriculum/blob/master/sms/)

## 🧠 Model Architecture

The model uses the following architecture:

- `TextVectorization` layer for converting raw text to integer sequences
- `Embedding` layer to learn word embeddings during training
- `GlobalAveragePooling1D` layer to reduce dimensionality
- `Dense` hidden layer with ReLU activation
- Output layer with a sigmoid activation for binary classification

## ⚙️ Tech Stack

| Component         | Tool/Library                 |
|------------------|------------------------------|
| Programming Lang | Python 3                     |
| Deep Learning    | TensorFlow, Keras            |
| NLP Tools        | TensorFlow TextVectorization |
| Data Handling    | Pandas, NumPy                |
| Visualization    | Matplotlib                   |
| Environment      | Google Colab / Jupyter       |

## 📈 Model Performance

- Validation Accuracy: ~95%+
- Loss: Tracked using training and validation curves

*Performance may vary slightly depending on training parameters.*

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sms-text-classification.git
cd sms-text-classification

## 🧠 Key Learnings
Applied natural language processing using TensorFlow pipelines.
Understood and implemented embedding-based deep learning models.
Learned how to use TextVectorization for efficient preprocessing.
Practiced evaluating model performance using accuracy/loss metrics.
Gained confidence working with external datasets and .tsv files.

## 🚧 Future Enhancements
Add Bidirectional LSTM or GRU layers for better sequence modeling.
Integrate pre-trained word embeddings like GloVe or FastText.
Create a Streamlit or Gradio app for interactive spam detection.
Deploy the model using Flask API or on Hugging Face Spaces.
Improve preprocessing pipeline with lemmatization, stopword removal, etc.














