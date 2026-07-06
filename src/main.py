import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE


def load_and_preprocess_data(filepath):
    """Carrega o dataset, trata valores nulos, padroniza e aplica SMOTE para balanceamento."""
    
    # 1. Carrega os dados
    df = pd.read_csv(filepath)
    
    # 2. Separa features (X) e alvo (y)
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # 3. Divide em treino e teste (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Tratamento de dados ausentes (Imputação pela mediana)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # 5. Padronização (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # 6. Balanceamento de Dados com SMOTE (Apenas no conjunto de treino!)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    """Treina o modelo, calcula métricas e registra tudo no MLflow."""
    
    # Inicia uma run no MLflow
    with mlflow.start_run(run_name=model_name):
        # Treinamento
        model.fit(X_train, y_train)
        
        # Predições no conjunto de teste
        y_pred = model.predict(X_test)
        
        # Cálculo das métricas
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log dos parâmetros e métricas no MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("smote_applied", True)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log do modelo de forma apropriada (sklearn ou xgboost)
        if "XGBoost" in model_name:
            mlflow.xgboost.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)
        
        print(f"--- {model_name} ---")
        print(f"Acurácia: {acc:.4f} | Precisão: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}\n")

if __name__ == "__main__":
    # Força o uso de um banco de dados local para evitar o bug de caminhos do Windows
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # Define o nome do experimento
    mlflow.set_experiment("Previsao_Potabilidade_Agua")
    
    print("Iniciando pré-processamento (Imputação, Padronização e SMOTE)...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/water_potability.csv")
    
    print("Iniciando treinamento dos modelos...\n")
    
    # --- MODELO 1: Regressão Logística (Baseline) ---
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    train_and_log_model("Regressao_Logistica", lr_model, X_train, X_test, y_train, y_test)
    
    # --- MODELO 2: Random Forest ---
    rf_model = RandomForestClassifier(random_state=42, n_estimators=150, max_depth=10)
    train_and_log_model("Random_Forest", rf_model, X_train, X_test, y_train, y_test)
    
    # --- MODELO 3: XGBoost ---
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=150, max_depth=6)
    train_and_log_model("XGBoost", xgb_model, X_train, X_test, y_train, y_test)
    
    print("Execução finalizada com sucesso! Rode 'mlflow ui' para analisar o benchmarking.")