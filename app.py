import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega os dados
df = pd.read_csv("filmes.csv")
vectorizer = TfidfVectorizer()
matriz_tfidf = vectorizer.fit_transform(df['descricao'])

# Função de recomendação
def recomendar(filme):
    if filme not in df['titulo'].values:
        return ["Filme não encontrado."]
    idx = df[df['titulo'] == filme].index[0]
    similaridades = cosine_similarity(matriz_tfidf[idx], matriz_tfidf).flatten()
    indices = similaridades.argsort()[::-1][1:4]
    return df.iloc[indices]['titulo'].tolist()

# Interface Streamlit
st.title("Recomendador de Filmes com IA")
filme_escolhido = st.selectbox("Escolha um filme:", df['titulo'])
if st.button("Recomendar"):
    resultado = recomendar(filme_escolhido)
    st.subheader("Recomendações:")
    for r in resultado:
        st.write("- ", r)

