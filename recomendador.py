import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)
    return df

def preparar_modelo(df):
    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(df['descricao'])
    return matriz_tfidf

def recomendar_filmes(filme_referencia, df, matriz_tfidf):
    if filme_referencia not in df['titulo'].values:
        return f'Filme "{filme_referencia}" não encontrado no banco de dados.'
    
    idx = df[df['titulo'] == filme_referencia].index[0]
    similaridades = cosine_similarity(matriz_tfidf[idx], matriz_tfidf).flatten()
    indices_similares = similaridades.argsort()[::-1][1:4]
    recomendacoes = df.iloc[indices_similares]['titulo'].tolist()
    return recomendacoes
