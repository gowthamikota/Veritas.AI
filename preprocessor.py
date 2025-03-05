import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
nltk.download("stopwords")
nltk.download("punkt")
dataset = load_dataset("json", data_files={
    "all_data": "HC3/all.jsonl",
    "finance": "HC3/finance.jsonl",
    "medicine": "HC3/medicine.jsonl",
    "open_qa": "HC3/open_qa.jsonl",
    "reddit_eli5": "HC3/reddit_eli5.jsonl",
    "wiki_csai": "HC3/wiki_csai.jsonl"
})

df = pd.DataFrame(dataset["all_data"])


df = df.rename(columns={"human_answers": "human_text", "chatgpt_answers": "ai_text"})


df["human_text"] = df["human_text"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
df["ai_text"] = df["ai_text"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))


df_human = pd.DataFrame({"text": df["human_text"], "label": 0})  # 0 = Human
df_ai = pd.DataFrame({"text": df["ai_text"], "label": 1})  # 1 = AI


df_combined = pd.concat([df_human, df_ai]).dropna().reset_index(drop=True)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):  
        return ""  
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words) 

df_combined["text"] = df_combined["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df_combined["text"], df_combined["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))  
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced")  
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "text_detection_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained and saved successfully!")
