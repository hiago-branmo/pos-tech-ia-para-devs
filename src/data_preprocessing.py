"""
Módulo de Pré-processamento de Dados para Predição de Diabetes
==============================================================

Este módulo contém todas as funções necessárias para carregar, limpar,
transformar e preparar os dados para modelagem de machine learning.

Funcionalidades principais:
- Carregamento e validação dos dados
- Tratamento de valores missing
- Encoding de variáveis categóricas
- Normalização e escalonamento
- Feature selection
- Divisão em conjuntos de treino/validação/teste
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class DiabetesDataProcessor:
    """
    Classe principal para processamento dos dados de diabetes.
    
    Attributes:
        scaler: StandardScaler para normalização
        label_encoders: Dict com encoders para variáveis categóricas
        feature_selector: SelectKBest para seleção de features
        selected_features: Lista das features selecionadas
    """
    
    def __init__(self, n_features: int = 15):
        """
        Inicializa o processador de dados.
        
        Args:
            n_features: Número de features a serem selecionadas
        """
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        self.selected_features = []
        self.n_features = n_features
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega os dados do arquivo CSV.
        
        Args:
            filepath: Caminho para o arquivo de dados
            
        Returns:
            DataFrame com os dados carregados
        """
        print("📊 Carregando dados...")
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Dados carregados com sucesso: {df.shape[0]} amostras, {df.shape[1]} features")
            return df
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            raise
            
    def explore_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Realiza exploração inicial dos dados.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            Dict com estatísticas descritivas
        """
        print("\n🔍 Explorando dados...")
        
        exploration_results = {
            'shape': df.shape,
            'missing_values': df.isnull().sum(),
            'data_types': df.dtypes,
            'target_distribution': df['Diabetes_binary'].value_counts() if 'Diabetes_binary' in df.columns else None,
            'numerical_stats': df.describe(),
        }
        
        print(f"📋 Shape: {exploration_results['shape']}")
        print(f"🔢 Missing values: {exploration_results['missing_values'].sum()}")
        
        if exploration_results['target_distribution'] is not None:
            print(f"🎯 Distribuição do target:")
            for value, count in exploration_results['target_distribution'].items():
                percentage = (count / len(df)) * 100
                print(f"   Classe {value}: {count} ({percentage:.1f}%)")
        
        return exploration_results
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Trata valores missing nos dados.
        
        Args:
            df: DataFrame com os dados
            strategy: Estratégia para imputação ('mean', 'median', 'mode')
            
        Returns:
            DataFrame com valores missing tratados
        """
        print(f"\n🔧 Tratando valores missing (estratégia: {strategy})...")
        
        df_clean = df.copy()
        missing_before = df_clean.isnull().sum().sum()
        
        if missing_before > 0:
            # Separar colunas numéricas e categóricas
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
            
            # Imputar valores numéricos
            if len(numeric_cols) > 0:
                if strategy in ['mean', 'median']:
                    imputer = SimpleImputer(strategy=strategy)
                    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
                    
            # Imputar valores categóricos
            if len(categorical_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                df_clean[categorical_cols] = imputer_cat.fit_transform(df_clean[categorical_cols])
            
            missing_after = df_clean.isnull().sum().sum()
            print(f"✅ Missing values: {missing_before} → {missing_after}")
        else:
            print("✅ Nenhum valor missing encontrado")
            
        return df_clean
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Codifica variáveis categóricas.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            DataFrame com variáveis codificadas
        """
        print("\n🔤 Codificando variáveis categóricas...")
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        # Excluir coluna target se presente
        if 'Diabetes_binary' in categorical_cols:
            categorical_cols = categorical_cols.drop('Diabetes_binary')
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                
        print(f"✅ {len(categorical_cols)} variáveis categóricas codificadas")
        return df_encoded
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List]:
        """
        Detecta outliers nos dados.
        
        Args:
            df: DataFrame com os dados
            method: Método de detecção ('iqr' ou 'zscore')
            
        Returns:
            Dict com outliers por coluna
        """
        print(f"\n🎯 Detectando outliers (método: {method})...")
        
        outliers_dict = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Excluir coluna target se presente
        if 'Diabetes_binary' in numeric_cols:
            numeric_cols = numeric_cols.drop('Diabetes_binary')
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > 3].index.tolist()
            
            if len(outliers) > 0:
                outliers_dict[col] = outliers
                
        total_outliers = sum(len(outliers) for outliers in outliers_dict.values())
        print(f"✅ {total_outliers} outliers detectados em {len(outliers_dict)} colunas")
        
        return outliers_dict
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'f_classif') -> pd.DataFrame:
        """
        Seleciona as melhores features para o modelo.
        
        Args:
            X: Features (variáveis independentes)
            y: Target (variável dependente)
            method: Método de seleção ('f_classif' ou 'mutual_info')
            
        Returns:
            DataFrame com features selecionadas
        """
        print(f"\n🎯 Selecionando {self.n_features} melhores features (método: {method})...")
        
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
        
        X_selected = self.feature_selector.fit_transform(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        print(f"✅ Features selecionadas: {len(self.selected_features)}")
        print(f"📋 Features: {', '.join(self.selected_features[:5])}...")
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normaliza as features usando StandardScaler.
        
        Args:
            X: Features para normalizar
            fit: Se deve ajustar o scaler (True para treino, False para teste)
            
        Returns:
            DataFrame com features normalizadas
        """
        print("\n📏 Normalizando features...")
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            print("✅ Scaler ajustado e dados normalizados")
        else:
            X_scaled = self.scaler.transform(X)
            print("✅ Dados normalizados usando scaler existente")
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, val_size: float = 0.2, 
                   random_state: int = 42, stratify: bool = True) -> Tuple:
        """
        Divide os dados em conjuntos de treino, validação e teste.
        
        Args:
            X: Features
            y: Target
            test_size: Proporção para teste
            val_size: Proporção para validação (do conjunto de treino)
            random_state: Seed para reprodutibilidade
            stratify: Se deve estratificar por classe
            
        Returns:
            Tuple com (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"\n✂️  Dividindo dados (treino: {1-test_size:.0%}, val: {val_size:.0%}, teste: {test_size:.0%})...")
        
        stratify_param = y if stratify else None
        
        # Primeira divisão: treino+val vs teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_param
        )
        
        # Segunda divisão: treino vs validação
        val_size_adjusted = val_size / (1 - test_size)  # Ajustar proporção
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=stratify_temp
        )
        
        print(f"✅ Divisão concluída:")
        print(f"   Treino: {X_train.shape[0]} amostras")
        print(f"   Validação: {X_val.shape[0]} amostras")
        print(f"   Teste: {X_test.shape[0]} amostras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def process_pipeline(self, filepath: str, **kwargs) -> Tuple:
        """
        Pipeline completo de processamento dos dados.
        
        Args:
            filepath: Caminho para o arquivo de dados
            **kwargs: Parâmetros opcionais para as etapas
            
        Returns:
            Tuple com dados processados e divididos
        """
        print("🚀 Iniciando pipeline de processamento de dados...")
        print("=" * 60)
        
        # 1. Carregar dados
        df = self.load_data(filepath)
        
        # 2. Exploração inicial
        exploration = self.explore_data(df)
        
        # 3. Tratar missing values
        df = self.handle_missing_values(df, strategy=kwargs.get('missing_strategy', 'median'))
        
        # 4. Codificar variáveis categóricas
        df = self.encode_categorical_variables(df)
        
        # 5. Detectar outliers
        outliers = self.detect_outliers(df, method=kwargs.get('outlier_method', 'iqr'))
        
        # 6. Separar features e target
        target_col = kwargs.get('target_column', 'diagnosed_diabetes')
        if target_col not in df.columns:
            raise ValueError(f"Coluna target '{target_col}' não encontrada nos dados")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 7. Seleção de features
        X_selected = self.select_features(X, y, method=kwargs.get('feature_method', 'f_classif'))
        
        # 8. Dividir dados
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X_selected, y,
            test_size=kwargs.get('test_size', 0.2),
            val_size=kwargs.get('val_size', 0.2),
            random_state=kwargs.get('random_state', 42)
        )
        
        # 9. Normalizar features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_val_scaled = self.scale_features(X_val, fit=False)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        print("\n" + "=" * 60)
        print("🎉 Pipeline de processamento concluído com sucesso!")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, exploration, outliers)


def create_summary_report(processor: DiabetesDataProcessor, 
                         exploration: Dict, outliers: Dict,
                         X_train: pd.DataFrame, y_train: pd.Series) -> str:
    """
    Cria relatório resumo do processamento.
    
    Args:
        processor: Instância do processador
        exploration: Resultados da exploração
        outliers: Outliers detectados
        X_train: Features de treino
        y_train: Target de treino
        
    Returns:
        String com o relatório
    """
    report = f"""
📊 RELATÓRIO DE PROCESSAMENTO DE DADOS - DIABETES
{'='*60}

🔍 DADOS ORIGINAIS:
• Shape: {exploration['shape'][0]:,} amostras × {exploration['shape'][1]} features
• Missing values: {exploration['missing_values'].sum():,}
• Distribuição do target:
  - Não diabético (0): {exploration['target_distribution'][0]:,} ({exploration['target_distribution'][0]/exploration['shape'][0]*100:.1f}%)
  - Diabético (1): {exploration['target_distribution'][1]:,} ({exploration['target_distribution'][1]/exploration['shape'][0]*100:.1f}%)

🔧 PROCESSAMENTO APLICADO:
• Tratamento de missing values: ✅
• Encoding de variáveis categóricas: ✅  
• Detecção de outliers: {len(outliers)} colunas com outliers
• Seleção de features: {len(processor.selected_features)} features selecionadas
• Normalização: StandardScaler aplicado

🎯 FEATURES SELECIONADAS:
{chr(10).join([f'• {feature}' for feature in processor.selected_features])}

📈 CONJUNTOS FINAIS:
• Treino: {len(X_train):,} amostras
• Features: {X_train.shape[1]}
• Balanceamento treino: {(y_train.value_counts()[0]/len(y_train)*100):.1f}% / {(y_train.value_counts()[1]/len(y_train)*100):.1f}%

✅ Dados prontos para modelagem!
"""
    return report


if __name__ == "__main__":
    # Exemplo de uso
    processor = DiabetesDataProcessor(n_features=15)
    
    # Processar dados (assumindo que o arquivo existe)
    try:
        results = processor.process_pipeline(
            filepath="../diabetes_dataset.csv",
            missing_strategy='median',
            outlier_method='iqr',
            feature_method='f_classif',
            test_size=0.2,
            val_size=0.2,
            random_state=42
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test, exploration, outliers = results
        
        # Gerar relatório
        report = create_summary_report(processor, exploration, outliers, X_train, y_train)
        print(report)
        
    except FileNotFoundError:
        print("❌ Arquivo de dados não encontrado. Execute este script do diretório correto.")