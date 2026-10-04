import pandas as pd

def limpar_dataset_spotify(caminho_original, caminho_novo):
    print("A carregar o dataset original...")
    df = pd.read_csv(caminho_original)

    
    colunas_desejadas = [
        'track_name', 'artists',
        'danceability', 'energy',
        'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 
        'track_genre'
    ]
    
    df_filtrado = df[colunas_desejadas].copy()
    
    #  forçar tipo numérico nas features de áudio
    # Se houver texto deslocado nestas colunas, o 'coerce' transforma o texto em NaN.
    colunas_audio = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    for col in colunas_audio:
        df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors='coerce')
        
    # forçar que o género NÃO seja numérico
    # Tentamos converter o género para número. Onde a conversão falhar (NaN),
    mascara_genero_valido = pd.to_numeric(df_filtrado['track_genre'], errors='coerce').isna()
    df_filtrado = df_filtrado[mascara_genero_valido]

    
    # limpa as linhas com dados realmente em falta e as linhas corrompidas 
    df_filtrado = df_filtrado.dropna()
    
    #  limpa de duplicatas
    df_filtrado = df_filtrado.drop_duplicates(subset=['track_name', 'artists'])

    #limita pega apenas os 15 gêneros mais frequentes. Ao comentar gera o csv com todos generos
    top_15 = df_filtrado['track_genre'].value_counts().nlargest(15).index
    df_filtrado = df_filtrado[df_filtrado['track_genre'].isin(top_15)]
    ######################################3
    
    print(f"A guardar o dataset limpo em {caminho_novo}...")
    df_filtrado.to_csv(caminho_novo, index=False)
    quantidade_generos = df_filtrado['track_genre'].nunique()

    
    print("\n--- Resumo da Limpeza ---")

    # Conta e exibe a quantidade de generos únicos
    
    print(f"\nExistem {quantidade_generos} generos musicais diferentes neste dataset.")
    print(f"Total de linhas iniciais: {len(df)}")
    print(f"Total de linhas finais:   {len(df_filtrado)}")

if __name__ == "__main__":
    arquivo_entrada = "./dataset.csv" 
    arquivo_saida = "spotify_limpo_limitado.csv"
    
    limpar_dataset_spotify(arquivo_entrada, arquivo_saida)