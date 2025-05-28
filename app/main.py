from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

app = FastAPI()

# Libera CORS para qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Aceita qualquer origem
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura o caminho para os dados do nltk (opcional, mas recomendado na Render)
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

# (Evite usar downloads na Render)
# nltk.download('stopwords')
# nltk.download('punkt')

class TextInput(BaseModel):
    text: str
    num_sentences: int = 3

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-záéíóúãõâêîôûç\s]', '', text)
    return text

@app.post("/summarize")
def summarize_text(input: TextInput) -> Dict[str, object]:
    text = preprocess_text(input.text)
    
    stop_words = set(stopwords.words('portuguese'))
    sentences = sent_tokenize(input.text, language='portuguese')
    num_sentences = min(input.num_sentences, len(sentences))
    
    if not sentences:
        return {
            "summary": "",
            "keywords": []
        }

    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
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
