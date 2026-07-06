# Projeto de Automação na Análise de Dados da Crise Global da Água Potável

**Instituição:** UNIVERSIDADE FEDERAL DO CEARÁ - CAMPUS SOBRAL  
**Curso:** ENGENHARIA DE COMPUTAÇÃO  
**Disciplina:** MÉTODOS COMPUTACIONAIS APLICADOS (2026.1) 
**Professor:** PROF. FERNANDO DANIEL DE OLIVEIRA MAYORGA

---

## 👥 Equipe
* VICTOR EMANUEL FONTENELE LIMA - 509539
* MARINA PAULA FONTENELE - 539022
* NICOLE SOUZA BATISTA - 541022
* PEDRO HENRIQUE MORAIS DA SILVA - 521461

---

## 🎯 Objetivo do Projeto
O objetivo deste projeto é construir e avaliar modelos preditivos de classificação supervisionada para determinar se uma amostra de água é potável ou não. Esta iniciativa visa servir como uma ferramenta de triagem automatizada e barata para populações vulnerávei, considerando que bilhões de pessoas sofrem com a falta de acesso à água segura para consumo, segundo a Organização Mundial da Saúde (OMS) e a UNICEF.

A metodologia do projeto inclui análise de amostras de água por meio de processos automatizados, análise estatística e o treinamento e teste de pelo menos três algoritmos de aprendizagem supervisionada para classificação.

---

## 📊 Base de Dados (Dataset)
Utilizamos o dataset **Water Potability**, hospedado no Kaggle (https://www.kaggle.com/datasets/adityakadiwal/water-potability). O dataset possui desafios técnicos, incluindo desbalanceamento e dados ausentes que exigiram tratamento prévio.

### Descrição dos Atributos
* **ph:** Medida do equilíbrio ácido-base da água (escala de 0 a 14).
* **Hardness:** Dureza da água, causada principalmente por sais de cálcio e magnésio (em mg/L).
* **Solids:** Sólidos Dissolvidos Totais (TDS), indicando o nível de mineralização (em ppm).
* **Chloramines:** Concentração de cloraminas, principais desinfetantes usados em sistemas públicos (em ppm).
* **Sulfate:** Concentração de sulfatos, substâncias abundantes na atmosfera e solo (em mg/L).
* **Conductivity:** Condutividade elétrica da água, ligada à presença de sólidos dissolvidos (μS/cm).
* **Organic_carbon:** Carbono orgânico total (TOC), mede o nível de matéria orgânica na amostra (em ppm).
* **Trihalomethanes:** Concentração de trihalometanos, subprodutos químicos gerados no tratamento da água (μg/L).
* **Turbidity:** Turbidez da água, que mede as propriedades de emissão de luz dependendo de matéria suspensa (NTU).
* **Potability (Atributo-alvo):** 1 se a água é potável (segura para consumo humano), 0 se não é potável.

---

## 🛠️ Tecnologias e Algoritmos Utilizados
* **Linguagem:** Python
* **Bibliotecas:** Pandas, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
* **Algoritmos de Classificação:** Regressão Logística, Random Forest e XGBoost.
* **Rastreamento e Comunicação de Resultados:** ML Flow (https://mlflow.org/).

---

## 📊 Análise Estatística e Exploração de Dados

Antes de treinar os modelos, realizamos uma etapa de Análise Exploratória de Dados (EDA) para compreender as características físico-químicas da água e mapear as dificuldades que os algoritmos de Machine Learning enfrentariam. Os principais insights são detalhados a seguir:

### 1. Desbalanceamento de Classes e Amostragem
A base de dados apresenta um desbalanceamento nítido entre amostras potáveis e não potáveis:
*   **Água Não Potável (Classe 0):** 1.998 amostras (61,0%)
*   **Água Potável (Classe 1):** 1.278 amostras (39,0%)

Esse desbalanceamento de classes prejudicaria o aprendizado de classificadores tradicionais, tornando-os tendenciosos em direção à classe majoritária (não potável). Para mitigar esse problema de forma científica, o pipeline implementado em `main.py` utiliza a técnica **SMOTE** (`Synthetic Minority Over-sampling Technique`), gerando dados sintéticos para equilibrar a proporção das classes antes do treinamento.

<img width="500" height="371" alt="distribuicao_classes" src="https://github.com/user-attachments/assets/6b29979f-f5dc-463d-9a12-9f92ba920e66" />

### 2. Tratamento de Dados Ausentes (Valores Nulos)
Detectamos que três variáveis cruciais possuem dados faltantes expressivos:
*   **Sulfate (Sulfato):** 781 valores ausentes (exige atenção, pois o sulfato é abundante no solo e indicador de mineralização).
*   **ph (pH):** 491 valores ausentes (essencial para medir o equilíbrio ácido-base).
*   **Trihalomethanes (Trihalometanos):** 162 valores ausentes (subprodutos químicos de tratamento).

Para evitar a perda de quase um terço do dataset por eliminação direta, o script utiliza a classe `SimpleImputer` com a estratégia de preenchimento pela **mediana** de cada coluna, preservando a integridade estatística das distribuições.

### 3. Distribuição do pH vs. Padrões de Segurança da OMS
O pH médio das amostras é de **7,08**, o que parece ideal à primeira vista. No entanto, a análise de dispersão mostra que uma quantidade massiva de amostras está fora da faixa de segurança regulamentada pela Organização Mundial da Saúde (OMS), que estabelece limites estritos entre **6,5 e 8,5** para consumo humano seguro. Amostras muito ácidas (pH < 6.5) ou muito alcalinas (pH > 8.5) representam riscos graves à saúde.

<img width="500" height="294" alt="distribuicao_ph" src="https://github.com/user-attachments/assets/e9b79775-40f4-474e-aa1f-1fe71871171b" />

### 4. Correlação Linear Desprezível (Relações Não-Lineares)
Ao calcular a matriz de correlação de Pearson entre todas as propriedades físico-químicas da água e a variável-alvo (`Potability`), observou-se que todos os coeficientes lineares são extremamente baixos (praticamente todos abaixo de **0.05**).

Isso prova empiricamente que **nenhum atributo isolado consegue determinar se a água é potável de forma linear**. Esse comportamento justifica a nossa abordagem técnica de benchmarking: modelos lineares simples como a *Regressão Logística* tendem a apresentar menor desempenho comparados a modelos de árvore baseados em ensembles não-lineares, como o *Random Forest* e o *XGBoost*, que conseguem capturar padrões complexos e interações profundas entre os dados químicos.

<img width="500" height="451" alt="heatmap_correlacao" src="https://github.com/user-attachments/assets/0ba8fa9b-49e5-426c-93e4-9b83d38c8bf6" />

---

## ⚙️ Instruções de Instalação e Configuração

Siga o passo a passo abaixo para instalar as dependências, executar o pipeline de treinamento e visualizar o monitoramento de experimentos no painel do MLflow [3].

### 1. Pré-requisitos
Certifique-se de ter o **Python 3.12** ou superior instalado em sua máquina.

### 2. Clonar o Repositório e Configurar o Ambiente
No seu terminal, execute os comandos abaixo para clonar o projeto e criar um ambiente virtual dedicado:

#### Clone o repositório da equipe:
git clone https://github.com/victoremanuelfont/metodos_computacionais_economia/
<br>cd Topicos-Projeto-Final

#### Crie um ambiente virtual (venv):
python -m venv venv

#### Ative o ambiente virtual
#### No Windows (Prompt de Comando):
venv\Scripts\activate

#### No Linux/macOS:
source venv/bin/activate

### 3. Instalar Dependências do Pipeline
#### Instale as bibliotecas necessárias para manipulação de dados, balanceamento, modelagem preditiva e tracking de experimentos:
pip install pandas numpy scikit-learn xgboost imbalanced-learn mlflow

### 4. Executar o Treinamento dos Modelos
O arquivo main.py está configurado para ler o arquivo water_potability.csv, realizar todo o pré-processamento (imputação de nulos, balanceamento com SMOTE e padronização com StandardScaler) e treinar três algoritmos supervisionados diferentes: Regressão Logística, Random Forest e XGBoost.

#### Para iniciar o processo, execute:
python main.py

### 5. Visualizar Experimentos no MLflow (Interface Gráfica)
Após a execução do script, todas as métricas geradas (Acurácia, F1-Score, Precisão e Recall), parâmetros dos modelos e os artefatos salvos estarão registrados.

Para abrir o dashboard interativo do MLflow e comparar o desempenho dos modelos, execute o comando abaixo no mesmo diretório:
<br>mlflow ui --backend-store-uri sqlite:///mlflow.db

Em seguida, abra o seu navegador de internet e acesse o endereço local gerado pelo MLflow:
<br>👉 http://localhost:5000

<br>Na interface web, você poderá:
* Visualizar as métricas de cada modelo preditivo sob o experimento Previsao_Potabilidade_Agua.
* Comparar graficamente as curvas de aprendizado e métricas de desempenho.
* Baixar ou exportar os binários dos modelos treinados para produção.

---

## 📹 Apresentação e Demonstração do Projeto
O vídeo de apresentação do projeto, contendo a explicação da metodologia de ciência de dados utilizada pela equipe, os resultados estatísticos e a demonstração ao vivo da execução do pipeline e do painel do MLflow, pode ser assistido no link abaixo:

<link youtube> https://www.youtube.com/watch?v=XrUt7laGOxA
