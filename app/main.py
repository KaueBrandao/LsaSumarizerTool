from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = FastAPI()

# Configuração do CORS
origins = [
    "http://localhost:3000",       # Permitir o frontend React local
    "http://localhost",            # Permitir localhost
    "http://localhost:5173",       # Caso esteja rodando no Vite
    "https://kauebrandao.github.io",  # Permitir o domínio do seu site no GitHub Pages
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Baixa os recursos do NLTK se ainda não tiver
nltk.download('stopwords')
nltk.download('punkt')

class TextInput(BaseModel):
    text: str
    num_sentences: int = 3  # Valor padrão de 3 sentenças

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

    # Divide o texto em sentenças
    sentences = sent_tokenize(input.text, language='portuguese')
    
    # Se o número de sentenças solicitadas for maior que o disponível, ajusta
    num_sentences = min(input.num_sentences, len(sentences))
    
    if not sentences:
        return {
            "summary": "",
            "keywords": []
        }

    # Converte set para lista para passar ao vectorizer
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calcula a pontuação de cada sentença com base na soma dos scores TF-IDF das palavras
    sentence_scores = []
    for i in range(len(sentences)):
        scores = tfidf_matrix[i].toarray()[0]
        total_score = sum(scores)
        sentence_scores.append((sentences[i], total_score))
    
    # Ordena as sentenças por pontuação e seleciona as mais relevantes
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    selected_sentences = [sentence for sentence, score in sorted_sentences[:num_sentences]]
    
    # Junta as sentenças para formar o resumo
    summary = " ".join(selected_sentences)
    
    # Extrai palavras-chave
    word_scores = {word: scores[idx] for idx, word in enumerate(feature_names) if len(word) > 2 and word.isalpha()}
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in sorted_words[:5]]

    return {
        "summary": summary,
        "keywords": keywords
    }
