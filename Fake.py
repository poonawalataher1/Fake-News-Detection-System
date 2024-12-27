import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud

# Tensorflow and Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#Reading the datasets
data_fake = pd.read_csv('/content/Fake.csv')
data_real = pd.read_csv('/content/True.csv')



# Checking the data
data_fake.head(10)
data_real.head(10)

# Data preprocessing and text cleaning
unknown_publishers = []
for index, row in enumerate(data_real.text.values):
  try:
    record = row.split(' - ',maxsplit = 1)
    record[1]
    assert(len(record[0]) < 260)
  except:
    unknown_publishers.append(index)


# Removing unwanted rows
data_real.drop(8970, axis = 0)

# Splitting publisher and text
publisher = []
temp_text = []
for index, row in enumerate(data_real.text.values):
  if index in unknown_publishers:
    temp_text.append(row)
    publisher.append('Unknown')
  else:
    record = row.split(' - ', maxsplit=1)
    publisher.append(record[0].strip())
    temp_text.append(record[1].strip())

data_real['publisher'] = publisher
data_real['text'] = temp_text



# Merging title and text, and lowering case
data_real['text'] = data_real['title'] + " " + data_real['text']
data_fake['text'] = data_fake['title'] + " " + data_fake['text']

data_real['text'] = data_real['text'].apply(lambda x: str(x).lower())
data_fake['text'] = data_fake['text'].apply(lambda x: str(x).lower())


# Assigning labels
data_real['class'] = 1
data_fake['class'] = 0


# Selecting relevant columns
data_real = data_real[['text', 'class']]
data_fake = data_fake[['text', 'class']]

# Concatenating both datasets
Data = pd.concat([data_real, data_fake], ignore_index=True)




# Tokenizing the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(Data['text'].values)

# Converting text to sequences and padding
X = tokenizer.texts_to_sequences(Data['text'].values)
X = pad_sequences(X, maxlen=700)



# Assigning target variable
y = Data['class'].values





# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1




#Loading pre-trained GloVe embeddings
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip

def load_glove(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.array(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings = load_glove('/content/glove.6B.300d.txt')





          # Preparing embedding matrix
          def get_glove_embedding_matrix(vocab, embedding_dim=300):
              embedding_matrix = np.zeros((vocab_size, embedding_dim))
              for word, i in vocab.items():
                  embedding_vector = glove_embeddings.get(word)
                  if embedding_vector is not None:
                      embedding_matrix[i] = embedding_vector
              return embedding_matrix

          embedding_vectors = get_glove_embedding_matrix(tokenizer.word_index,
                                                        embedding_dim=300)





# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)



# Building the model with Dropout layers
model = Sequential()
model.add(Embedding(vocab_size, output_dim=300, weights=[embedding_vectors],
                    input_length=1000, trainable=True))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dense(1, activation='sigmoid'))



# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()



# Defining class weights for handling imbalance (adjust as needed)
class_weights = {0: 1.0, 1: 1.0}


# Training the model with class weights
# model.fit(X_train, y_train, validation_split=0.3, epochs=6, class_weight=class_weights)
model.fit(X_train, y_train, validation_split=0.7, epochs=7, class_weight=class_weights)  # Increase epochs for better learning





# Evaluating model
y_pred = (model.predict(X_test) >= 0.7).astype(int)  # Adjust threshold as needed



# Metrics and performance evaluation
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))




from sklearn.metrics import confusion_matrix, classification_report

y_pred = (model.predict(X_test) >= 0.9).astype(int)  # Adjust threshold here too
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_news(news_text, tokenizer, model, maxlen=1000, threshold=0.9):
    """
    This function takes in a news text, tokenizes it, pads it to the required length,
    and predicts whether it is real or fake news using the trained model.
    Args:
    news_text (str): The news article to be checked.
    tokenizer (Tokenizer): The tokenizer used for training.
    model (Sequential): The trained model.
    maxlen (int): The maximum length for padding.
    threshold (float): The threshold for classifying the news as real or fake.

    Returns:
    str: The prediction result - 'Real News' or 'Fake News'.
    """
    # Preprocess the input text
    news_seq = tokenizer.texts_to_sequences([news_text])
    news_padded = pad_sequences(news_seq, maxlen=maxlen)

    # Predict using the model
    prediction = model.predict(news_padded)[0][0]  # Get the predicted probability

    # Output the result based on the threshold
    if prediction >= threshold:
        return "Real News"
    else:
        return "Fake News"

# Example usage: Taking user input and making a prediction
user_news = input("Enter the news you want to check: ")
result = predict_news(user_news, tokenizer, model)
print(f"Prediction: {result}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake News', 'Real News'], yticklabels=['Fake News', 'Real News'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


from sklearn.metrics import roc_curve, auc

# Predicting probabilities for ROC curve
y_pred_prob = model.predict(X_test).ravel()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import precision_recall_curve

# Predicting probabilities for Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plotting Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# # Example: If y_pred is a NumPy array, convert it to a Pandas Series
# y_pred_series = pd.Series(y_pred.flatten(), name="Predictions")

# # Plotting class distribution
# plt.figure(figsize=(8, 6))
# sns.countplot(x=y_pred_series)
# plt.title("Distribution of Predicted Classes")
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.show()

# Assuming the output is in probability format, convert to binary class labels
y_pred_class = (y_pred >= 0.5).astype(int)  # Threshold at 0.5 for binary classification

# Convert to Pandas Series if necessary
y_pred_class_series = pd.Series(y_pred_class.flatten(), name="Predicted Class")

# Plotting class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=y_pred_class_series)
plt.title("Distribution of Predicted Classes")
plt.xlabel("Class (0: Fake News, 1: Real News)")
plt.ylabel("Count")
plt.show()


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter # Import Counter from collections module

# Download stopwords if not already present
nltk.download('stopwords')
nltk.download('punkt')

# Defining English stop words
stop_words = set(stopwords.words('english'))

# Tokenizing and removing stop words
filtered_words = []
for text in Data['text']:
    tokens = word_tokenize(text)
    filtered_words.extend([word for word in tokens if word.isalpha() and word not in stop_words])

# Getting the 20 most common words after removing stop words
word_counts = Counter(filtered_words).most_common(20) # Now Counter should be recognized

# Extracting words and their frequencies
words = [item[0] for item in word_counts]
counts = [item[1] for item in word_counts]

# Plotting the top 20 words
plt.figure(figsize=(12, 6))
sns.barplot(x=counts, y=words, palette='inferno')
plt.title("Top 20 Most Frequent Words in the Dataset (Excluding Stop Words)")
plt.xlabel("Word Count")
plt.ylabel("Words")
plt.show()

