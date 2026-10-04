# Classificação Semi-Supervisionada de Gêneros Musicais usando Atributos do Spotify

Este projeto aplica técnicas de **Aprendizado Semi-Supervisionado** (utilizando o algoritmo *Label Spreading*) para prever gêneros de músicas com base em atributos acústicos extraídos do Spotify (como *danceability*, *energy*, *acousticness*, *tempo*, entre outros).

O principal objetivo é demonstrar como a propagação de rótulos em grafos de vizinhança pode classificar um grande volume de dados não rotulados utilizando apenas uma pequena fração de amostras conhecidas.

---

##  Tecnologias Utilizadas

* **Python 3.x**
* **Pandas & NumPy** (Manipulação e estruturação de dados)
* **Scikit-Learn** (Pré-processamento, modelo *Label Spreading* e métricas de avaliação)
* **Matplotlib** (Geração de curvas de validação e gráficos analíticos)

---

##  Estrutura do Projeto

* `dataset.csv`: Base de dados original contendo as faixas do Spotify e seus atributos.
* `spotify_limpo_limitado.csv`: Dataset higienizado e filtrado para os gêneros mais frequentes.
* `spotify_limpo.csv`: Dataset completo contendo todos os 113 gêneros originais.
* `limpar_dataset.py`: Script responsável pela limpeza, tratamento de desalinhamentos e remoção de duplicatas.
* `label_spreading.py`: Script principal que executa o mascaramento (simulação semi-supervisionada), o treinamento do *Label Spreading* e a extração de métricas (Acurácia e F1-Score).


---
