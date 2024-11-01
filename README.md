# NLP-Detecting-Spam-Emails


This project aims to build a machine learning model that can accurately classify SMS messages as either spam or non-spam. We will explore various Natural Language Processing (NLP) techniques to prepare the text data and train a model to effectively detect spam. This is an essential tool for filtering out unwanted and potentially malicious messages.


The dataset used for this project consists of a collection of SMS messages labeled as either "spam" or "non-spam." It includes the following columns:

# Dataset
- S.No: Serial number
- Message_body: The text content of the SMS message
- Label: The classification of the message as "spam" or "non-spam"

  
# Project Phases

The project was conducted in the following phases:

1. **Data Acquisition and Exploration** 📊:
   - Loading the SMS spam dataset (SMS_train.csv and SMS_test.csv).
   - Performing exploratory data analysis (EDA) to understand the distribution of classes and features.
   - Visualizing data using histograms and word clouds.

2. **Data Preprocessing** 🧹:
   - Converting text to lowercase.
   - Removing punctuation and special characters.
   - Tokenizing messages into individual words.
   - Addressing class imbalance using techniques like RandomOverSampler.

3. **Feature Extraction** 🧮:
   - Utilizing TF-IDF (Term Frequency-Inverse Document Frequency) to convert text messages into numerical features.

4. **Model Training and Evaluation** 🧠:
   - Training a Multinomial Naive Bayes classifier on the extracted features.
   - Evaluating the model's performance using metrics like accuracy, precision, recall, and F1-score.

5. **Testing on Unseen Data** 🧪:
   - Applying the trained model to a separate test dataset (SMS_test.csv) to assess its generalization ability.
   - Analyzing the model's performance on unseen data.


The Multinomial Naive Bayes model achieved impressive results on both the training and test datasets:

- **High Accuracy:** The model achieved high accuracy in classifying messages as spam or non-spam.
- **Excellent Precision:** The model demonstrated a high precision, meaning that it correctly identified spam messages with minimal false positives.
- **Good Recall:** The model also achieved a good recall score, indicating its ability to capture most of the actual spam messages.
- **Balanced F1-Score:** The F1-score, which balances precision and recall, was also high, reflecting a well-rounded model performance.

## Future Improvements 🚀

* **Enhancing Recall:** Further exploration of techniques to improve recall and catch more spam messages without significantly increasing false positives.
* **Contextual Understanding:** Incorporating methods to better understand the context of messages, such as using n-grams or word embeddings.
* **Model Fine-tuning:** Optimizing model hyperparameters to enhance overall performance.
* **Alternative Models:** Exploring other models like Support Vector Machines (SVM) or Logistic Regression for potential performance improvements.
  
