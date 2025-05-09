from recomendador import carregar_dados, preparar_modelo, recomendar_filmes

def main():
    df = carregar_dados('filmes.csv')
    matriz_tfidf = preparar_modelo(df)
    
    filme = input("Digite um filme que você gosta: ")
    recomendacoes = recomendar_filmes(filme, df, matriz_tfidf)
    
    print("\nRecomendações para você:")
    if isinstance(recomendacoes, list):
        for r in recomendacoes:
            print(f"- {r}")
    else:
        print(recomendacoes)

if __name__ == "__main__":
    main()
