from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = FastAPI()

# Configuração do CORS
origins = [
    "http://localhost:3000",       # Permitir o frontend React local
    "http://localhost",            # Permitir localhost
    "http://localhost:5173",       # Caso esteja rodando no Vite
    "https://kauebrandao.github.io",  # Permitir o domínio do seu site no GitHub Pages
    # Adicione outros domínios conforme necessário
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Permitir origens específicas
    allow_credentials=True,
    allow_methods=["*"],            # Permitir qualquer método (GET, POST, etc.)
    allow_headers=["*"],            # Permitir qualquer cabeçalho
)

# Baixa as stopwords do NLTK se ainda não tiver
nltk.download('stopwords')

class TextInput(BaseModel):
    text: str

def preprocess_text(text: str) -> str:
    # Deixa tudo minúsculo e remove caracteres que não são letras ou espaços
    text = text.lower()
    text = re.sub(r'[^a-záéíóúãõâêîôûç\s]', '', text)
    return text

@app.post("/summarize")
def summarize_text(input: TextInput) -> Dict[str, object]:
    text = preprocess_text(input.text)
    
    # Stopwords em português
    stop_words = set(stopwords.words('portuguese'))

    def filter_word(word: str) -> bool:
        # Remove stopwords, palavras pequenas e não alfabéticas
        return (word not in stop_words) and (len(word) > 2) and word.isalpha()

    # Converte set para lista para passar ao vectorizer
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    word_scores = {word: scores[idx] for idx, word in enumerate(feature_names) if filter_word(word)}

    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    keywords = [word for word, score in sorted_words[:5]]

    summary = input.text[:500]

    return {
        "summary": summary,
        "keywords": keywords
    }