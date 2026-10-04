import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score, f1_score, classification_report


def executar_label_spreading(caminho_csv, tamanho_amostra):
    print(f"=== INICIANDO MODELO SEMI-SUPERVISIONADO ===")
    
    # 1. Carregamento e Amostragem
    df = pd.read_csv(caminho_csv)
    
    if len(df) > tamanho_amostra:
        print(f"Reduzindo dataset de {len(df)} para {tamanho_amostra} ")
        df = df.sample(n=tamanho_amostra, random_state=42)
        
    features_audio = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    X = df[features_audio].values
    y_texto = df['track_genre'].values
    
    # 2. Pré-processamento
    print("[1/4] Aplicando StandardScaler e LabelEncoder...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    encoder = LabelEncoder()
    y_codificado = encoder.fit_transform(y_texto)
    
    # 3. Mascaramento (Ocultando 80% dos rótulos)
    print("[2/4] Mascarando 80% dos dados (simulando cenário real)...")
    rng = np.random.RandomState(42)
    mascara_oculta = rng.rand(len(y_codificado)) < 0.7
    
    y_mascarado = np.copy(y_codificado)
    y_mascarado[mascara_oculta] = -1  # -1 --> genero "não rotulado"
    
    qtd_conhecida = np.sum(~mascara_oculta)
    qtd_oculta = np.sum(mascara_oculta)
    print(f" - Músicas com gênero conhecido (30%): {qtd_conhecida}")
    print(f" - Músicas para adivinhar (70%): {qtd_oculta}")
    
    # 4. Treinamento
    print("\n[3/4] Treinando o Label Spreading (propagando pelo grafo)...")

    # kernel 'knn' conecta cada música apenas aos seus 7 vizinhos mais próximos, 
    modelo_ls = LabelSpreading(kernel='knn', n_neighbors=50, alpha=0.3, max_iter=40)



    
    inicio = time.time()
    modelo_ls.fit(X_scaled, y_mascarado)
    fim = time.time()
    
    print(f"Treinamento concluído em {fim - inicio:.2f} segundos.")
    
    # 5. Avaliação (Estritamente nos dados ocultos)
    print("[4/4] Avaliando métricas nos dados que estavam ocultos...")
    y_pred_total = modelo_ls.transduction_
    
    y_real_oculto = y_codificado[mascara_oculta]
    y_pred_oculto = y_pred_total[mascara_oculta]
    
    acc = accuracy_score(y_real_oculto, y_pred_oculto)
    f1_macro = f1_score(y_real_oculto, y_pred_oculto, average='macro', zero_division=0)
    f1_micro = f1_score(y_real_oculto, y_pred_oculto, average='micro', zero_division=0)
    
    print("\n" + "="*40)
    print(" RESULTADOS DA PROPAGAÇÃO (70% Ocultos)")
    print("="*40)
    print(f"Acurácia Global: {acc:.4f}")
    print(f"Micro F1-Score:  {f1_micro:.4f}")
    print(f"Macro F1-Score:  {f1_macro:.4f}")

    print("\n--- Relatório Detalhado por Género ---")
    # O encoder.classes_ devolve os nomes originais dos generos na ordem correta
    print(classification_report(y_real_oculto, y_pred_oculto, target_names=encoder.classes_, zero_division=0))

    
    return modelo_ls, encoder

if __name__ == "__main__":
    executar_label_spreading("spotify_limpo_limitado.csv", tamanho_amostra=14000)