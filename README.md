# Email_spam_classifier

[![Open In Colab]([https://colab.research.google.com/assets/colab-badge.svg](https://colab.research.google.com/drive/18h6YMcVhVmgjl5L_YBQ8uKBjWH2PHzeR?usp=sharing))]

This project demonstrates a simple spam classifier using Natural Language Processing (NLP) and machine learning techniques. It aims to classify SMS messages as either "spam" or "ham" (not spam).

## Project Overview

* **Goal:** Create a model to accurately distinguish between spam and non-spam SMS messages.
* **Techniques:**
    - Text Preprocessing (Removing punctuation, stop words)
    - Bag of Words (BoW)
    - TF-IDF (Term Frequency-Inverse Document Frequency)
    - Naive Bayes Classifier
* **Dataset:**  SMSSpamCollection
* **Libraries:** Pandas, scikit-learn, NLTK

## How to Run

1. **Clone:** Clone this repository to your local machine.
2. **Install Dependencies:** `pip install -r requirements.txt` 
3. **Run the Script:** `python spam_classifier.py`

## Results

The model achieves an accuracy of over 97% on the test dataset, demonstrating its effectiveness in classifying spam messages.

## Acknowledgments

Special thanks to Sameer for his guidance and mentorship in this project.

## License

This project is licensed under the MIT License - see the License.md file for details.
