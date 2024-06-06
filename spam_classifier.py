import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the Data
messages = pd.read_csv("SMSSpamCollection", sep='\t', names=['label', 'message'])

# 2. Add a Length Column for EDA (Optional)
messages['length'] = messages['message'].apply(len)

# 3. Define Text Processing Function
def text_process(mess):
    """
    This function cleans and preprocesses text data.
    """
    import string
    from nltk.corpus import stopwords
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# 4. Split Data into Training and Testing Sets
msg_train, msg_test, label_train, label_test = train_test_split(
    messages['message'], messages['label'], test_size=0.2, random_state=42
)

# 5. Build and Train the Model Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # Bag-of-Words transformation
    ('tfidf', TfidfTransformer()),                   # TF-IDF transformation
    ('classifier', MultinomialNB()),                 # Naive Bayes classifier
])

pipeline.fit(msg_train, label_train)

# 6. Make Predictions on Test Set
predictions = pipeline.predict(msg_test)

# 7. Evaluate the Model
print(classification_report(label_test, predictions))
print(confusion_matrix(label_test, predictions))
