#📱SMS Spam Text Classification

A deep learning-based binary text classifier that identifies whether an incoming SMS message is spam or ham (not spam). This project leverages TensorFlow and natural language processing (NLP) techniques to create a robust and scalable message filtering system.

#🔍 Problem Statement
SMS spam messages are a widespread issue, wasting users’ time and posing security threats. The goal of this project is to build a neural network model that can automatically classify SMS messages as either spam or ham, based on their content. This is a classic binary classification problem.

#🚀 Project Overview
✅ Collected and loaded a labeled SMS dataset containing spam and ham messages.
✂️ Preprocessed text using TensorFlow's built-in TextVectorization layer.
🧠 Built a deep learning model using TensorFlow Keras.
📊 Trained, validated, and tested the model on real SMS data.
📈 Evaluated the model with appropriate metrics (accuracy, loss).
🔍 Visualized training performance using matplotlib.

#📂 Dataset
Format: .tsv (Tab-separated values)
Size: ~5.5k SMS messages
Labels: spam and ham
Source: FreeCodeCamp GitHub - SMS Dataset

#🧠 Model Architecture
The model is a Sequential Keras model consisting of:
✅ TextVectorization Layer – converts raw strings into integer sequences
✅ Embedding Layer – learns word embeddings during training
✅ GlobalAveragePooling1D – reduces sequence to a fixed-size vector
✅ Dense Hidden Layer – with ReLU activation
✅ Output Layer – single unit with sigmoid activation for binary classification

Note: You can also experiment with Bidirectional LSTMs or GRUs for potentially better performance on sequential data.

#⚙️ Tech Stack
Component	Tool/Library
Language	Python 3
Deep Learning	TensorFlow, Keras
NLP Preprocessing	TensorFlow TextVectorization
Visualization	Matplotlib
Data Handling	Pandas, NumPy
Environment	Google Colab / Jupyter

#📈 Model Performance
Accuracy: ~95%+ on validation set
Loss: Monitored via training/validation graphs
Model performance may vary depending on training epochs and vectorization parameters.

#🛠 How to Run
bash
Copy
Edit
# Step 1: Clone the repository
git clone https://github.com/yourusername/sms-text-classification.git
cd sms-text-classification

# Step 2: Open in Google Colab or Jupyter
# Recommended: Google Colab (handles tf and pip installs smoothly)
Or click the badge below to run it directly in Colab:


#📌 Key Learnings
Applied NLP preprocessing using TensorFlow pipelines.
Understood and implemented embedding-based models for text classification.
Improved understanding of vectorization, tokenization, and performance tuning.
Practiced working with external .tsv datasets in cloud environments like Colab.

#🧠 Future Enhancements
Add attention or transformer-based layers.
Experiment with pre-trained embeddings (GloVe, FastText).
Create a Streamlit or Gradio demo for user input classification.
Deploy the model via Flask or Hugging Face Spaces.

