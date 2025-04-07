# Customer Support Analysis System

This project implements a machine learning-based system to analyze customer support conversations, predict key attributes (issue type, sentiment, and priority), and recommend solutions based on historical data. It is designed to streamline customer support operations by automating the analysis of customer inquiries.

---

## Features

- **Text Preprocessing**: Includes tokenization, stopword removal, and lemmatization using NLTK
- **Conversation Parsing**: Extracts metadata (e.g., conversation ID, category, sentiment, priority) and text from raw conversation logs
- **TF-IDF Feature Extraction**: Converts preprocessed text into numerical features for classification
- **Machine Learning Models**:
  - Random Forest classifiers for predicting issue type, sentiment, and priority
- **Solution Mapping**: Matches predicted attributes to potential solutions using historical ticket data
- **New Inquiry Analysis**: Predicts attributes for new customer inquiries and provides recommended solutions

---

## Installation

1. Clone the repository:
git clone https://github.com/chicknugget/Accenture-Hackathon-.git

cd Accenture-Hackathon-


2. Install required Python libraries:
pip install -r requirements.txt

3. Download necessary NLTK resources:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


---

## Usage

### 1. Preprocess Text
The `preprocess_text` function handles text preprocessing:

def preprocess_text(text):
# Convert to lowercase
text = text.lower()
# Tokenize text
tokens = nltk.word_tokenize(text)
# Remove stopwords and short tokens
tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
# Lemmatize tokens
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]
return ' '.join(tokens)


### 2. Parse Conversations
The `parse_conversation` function extracts metadata:

def parse_conversation(conversation):
conversation_id = conversation['id']
category = conversation.get('category', 'Unknown')
messages = [msg['content'] for msg in conversation['messages']]
full_text = ' '.join(messages)
return {
'id': conversation_id,
'category': category,
'text': full_text
}


### 3. Analyze New Inquiries
Example usage for new customer inquiries:

new_inquiry = "The app crashes when I try to upload photos"
results = analyze_text_nlp(
new_inquiry,
issue_classifier,
sentiment_classifier,
priority_classifier,
tfidf_vectorizer,
issue_solution_mapping
)

print(f"Predicted Issue: {results}")
print(f"Sentiment: {results}")
print(f"Priority: {results}")
print("Recommended Solutions:")
for solution in results:
print(f"- {solution}")


---

## File Structure

├── AccHack.ipynb # Main application logic

├── Historical_ticket_data.csv # Training dataset

├── requirements.txt # Dependency list

└── README.md # This documentation


---

## Dependencies

pandas==1.3.4
scikit-learn==0.24.2
nltk==3.6.2
python-dotenv==0.19.0

5. Open a pull request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
