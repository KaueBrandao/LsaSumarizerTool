from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str
    num_sentences: int = 3

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-záéíóúãõâêîôûç\s]', '', text)
    return text

# Tokenização de frases simples (substitui o sent_tokenize)
def split_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# Stopwords básicas em português
BASIC_STOPWORDS = set([
    "a", "o", "e", "é", "de", "do", "da", "em", "um", "uma", "que", "com", "por", "para",
    "os", "as", "no", "na", "se", "ao", "dos", "das", "ou", "mais", "mas", "como"
])

@app.post("/summarize")
def summarize_text(input: TextInput) -> Dict[str, object]:
    sentences = split_sentences(input.text)
    num_sentences = min(input.num_sentences, len(sentences))

    if not sentences:
        return {"summary": "", "keywords": []}

    vectorizer = TfidfVectorizer(stop_words=list(BASIC_STOPWORDS))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    sentence_scores = []
    for i in range(len(sentences)):
        scores = tfidf_matrix[i].toarray()[0]
        total_score = sum(scores)
        sentence_scores.append((sentences[i], total_score))

    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    selected_sentences = [sentence for sentence, score in sorted_sentences[:num_sentences]]

    summary = " ".join(selected_sentences)

    word_scores = {word: scores[idx] for idx, word in enumerate(feature_names) if len(word) > 2 and word.isalpha()}
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in sorted_words[:5]]

    return {
        "summary": summary,
        "keywords": keywords
    }
