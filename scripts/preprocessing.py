"""
Pré-processamento de Dados - Dataset de Diabetes
Tech Challenge IADT - Fase 1

Este módulo contém funções para:
- Limpeza de dados (valores ausentes, inconsistências)
- Pipeline de pré-processamento de dados
- Encoding de variáveis categóricas
- Normalização/Padronização de variáveis numéricas
- Análise de correlação
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DiabetesDataPreprocessor:
    """
    Classe para pré-processamento do dataset de diabetes
    """

    def __init__(self, data_path='diabetes_dataset.csv'):
        """
        Inicializa o preprocessador

        Args:
            data_path (str): Caminho para o arquivo CSV
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None

        # Definir colunas por tipo
        self.categorical_cols = [
            'gender', 'ethnicity', 'education_level', 'income_level',
            'employment_status', 'smoking_status', 'diabetes_stage'
        ]

        self.numerical_cols = [
            'age', 'bmi', 'glucose_fasting', 'glucose_postprandial',
            'hba1c', 'insulin_level', 'diabetes_risk_score',
            'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
            'triglycerides', 'systolic_bp', 'diastolic_bp',
            'physical_activity_minutes_per_week', 'sleep_hours_per_day',
            'diet_score', 'alcohol_consumption_per_week', 'screen_time_hours_per_day',
            'family_history_diabetes', 'hypertension_history', 'cardiovascular_history',
            'waist_to_hip_ratio', 'heart_rate'
        ]

        self.target_col = 'diagnosed_diabetes'


    def load_data(self):
        """
        Carrega os dados do arquivo CSV
        """
        print("=" * 80)
        print("CARREGANDO DADOS")
        print("=" * 80)

        self.df = pd.read_csv(self.data_path)

        print(f"\nDataset carregado com sucesso!")
        print(f"Dimensões: {self.df.shape[0]} linhas x {self.df.shape[1]} colunas")
        print(f"\nPrimeiras 5 linhas:")
        print(self.df.head())

        return self


    def check_data_quality(self):
        """
        Verifica qualidade dos dados: valores ausentes, duplicatas, inconsistências
        """
        print("\n" + "=" * 80)
        print("ANÁLISE DE QUALIDADE DOS DADOS")
        print("=" * 80)

        # 1. Valores ausentes
        print("\n1. VALORES AUSENTES:")
        print("-" * 80)
        missing_values = self.df.isnull().sum()
        missing_pct = (missing_values / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Coluna': missing_values.index,
            'Valores Ausentes': missing_values.values,
            'Percentual (%)': missing_pct.values
        })
        missing_df = missing_df[missing_df['Valores Ausentes'] > 0].sort_values('Valores Ausentes', ascending=False)

        if len(missing_df) > 0:
            print("\nColunas com valores ausentes:")
            print(missing_df.to_string(index=False))
            print(f"\nTotal de colunas com valores ausentes: {len(missing_df)}")
        else:
            print("Nenhum valor ausente encontrado!")

        # 2. Duplicatas
        print("\n2. DUPLICATAS:")
        print("-" * 80)
        duplicates = self.df.duplicated().sum()
        print(f"Número de linhas duplicadas: {duplicates}")
        if duplicates > 0:
            print(f"Percentual de duplicatas: {(duplicates/len(self.df))*100:.2f}%")

        # 3. Tipos de dados
        print("\n3. TIPOS DE DADOS:")
        print("-" * 80)
        print(self.df.dtypes)

        # 4. Estatísticas básicas de variáveis numéricas
        print("\n4. ESTATÍSTICAS BÁSICAS (Variáveis Numéricas):")
        print("-" * 80)
        print(self.df[self.numerical_cols].describe())

        # 5. Valores únicos de variáveis categóricas
        print("\n5. VALORES ÚNICOS (Variáveis Categóricas):")
        print("-" * 80)
        for col in self.categorical_cols:
            unique_vals = self.df[col].nunique()
            print(f"{col:25s}: {unique_vals} valores únicos -> {self.df[col].unique()}")

        # 6. Verificar valores inconsistentes (outliers extremos)
        print("\n6. VERIFICAÇÃO DE VALORES INCONSISTENTES:")
        print("-" * 80)

        inconsistencies = []

        # Verificar idade
        if (self.df['age'] < 0).any() or (self.df['age'] > 120).any():
            inconsistencies.append("Idade com valores fora do esperado (< 0 ou > 120)")

        # Verificar BMI
        if (self.df['bmi'] < 10).any() or (self.df['bmi'] > 60).any():
            inconsistencies.append("BMI com valores extremos (< 10 ou > 60)")

        # Verificar glicose
        if (self.df['glucose_fasting'] < 50).any() or (self.df['glucose_fasting'] > 300).any():
            inconsistencies.append("Glicose em jejum com valores extremos")

        # Verificar HbA1c
        if (self.df['hba1c'] < 3).any() or (self.df['hba1c'] > 15).any():
            inconsistencies.append("HbA1c com valores extremos")

        # Verificar pressão arterial
        if (self.df['systolic_bp'] < 70).any() or (self.df['systolic_bp'] > 250).any():
            inconsistencies.append("Pressão sistólica com valores extremos")

        if (self.df['diastolic_bp'] < 40).any() or (self.df['diastolic_bp'] > 150).any():
            inconsistencies.append("Pressão diastólica com valores extremos")

        if len(inconsistencies) > 0:
            print("ATENÇÃO: Possíveis inconsistências detectadas:")
            for i, issue in enumerate(inconsistencies, 1):
                print(f"  {i}. {issue}")
        else:
            print("Nenhuma inconsistência grave detectada!")

        # 7. Distribuição da variável alvo
        print("\n7. DISTRIBUIÇÃO DA VARIÁVEL ALVO (diagnosed_diabetes):")
        print("-" * 80)
        target_dist = self.df[self.target_col].value_counts()
        target_pct = self.df[self.target_col].value_counts(normalize=True) * 100
        print(f"Classe 0 (Sem diabetes): {target_dist[0]} ({target_pct[0]:.2f}%)")
        print(f"Classe 1 (Com diabetes): {target_dist[1]} ({target_pct[1]:.2f}%)")

        balance_ratio = min(target_dist) / max(target_dist)
        print(f"\nBalanceamento: {balance_ratio:.2f}")
        if balance_ratio < 0.5:
            print("ATENÇÃO: Dataset desbalanceado! Considerar técnicas de balanceamento.")
        else:
            print("Dataset relativamente balanceado.")

        return self


    def clean_data(self):
        """
        Realiza limpeza dos dados
        """
        print("\n" + "=" * 80)
        print("LIMPEZA DOS DADOS")
        print("=" * 80)

        # Fazer cópia para não modificar dados originais
        self.df_processed = self.df.copy()

        # 1. Remover duplicatas
        print("\n1. Removendo duplicatas...")
        initial_shape = self.df_processed.shape[0]
        self.df_processed = self.df_processed.drop_duplicates()
        final_shape = self.df_processed.shape[0]
        print(f"   Linhas removidas: {initial_shape - final_shape}")

        # 2. Tratar valores ausentes
        print("\n2. Tratando valores ausentes...")
        missing_before = self.df_processed.isnull().sum().sum()

        if missing_before > 0:
            # Para numéricas: preencher com mediana
            for col in self.numerical_cols:
                if self.df_processed[col].isnull().any():
                    median_val = self.df_processed[col].median()
                    self.df_processed[col].fillna(median_val, inplace=True)
                    print(f"   - {col}: preenchido com mediana ({median_val:.2f})")

            # Para categóricas: preencher com moda
            for col in self.categorical_cols:
                if self.df_processed[col].isnull().any():
                    mode_val = self.df_processed[col].mode()[0]
                    self.df_processed[col].fillna(mode_val, inplace=True)
                    print(f"   - {col}: preenchido com moda ({mode_val})")
        else:
            print("   Nenhum valor ausente encontrado!")

        missing_after = self.df_processed.isnull().sum().sum()
        print(f"   Valores ausentes antes: {missing_before}")
        print(f"   Valores ausentes depois: {missing_after}")

        # 3. Tratar outliers extremos (opcional - mantendo por enquanto)
        print("\n3. Análise de outliers...")
        print("   (Mantendo outliers para análise posterior - podem ser casos reais)")

        # 4. Validar tipos de dados
        print("\n4. Validando tipos de dados...")
        # Garantir que variáveis numéricas estão no tipo correto
        for col in self.numerical_cols:
            if self.df_processed[col].dtype == 'object':
                self.df_processed[col] = pd.to_numeric(self.df_processed[col], errors='coerce')
                print(f"   - {col}: convertido para numérico")

        print("\n✓ Limpeza concluída!")
        print(f"Dataset limpo: {self.df_processed.shape[0]} linhas x {self.df_processed.shape[1]} colunas")

        return self


    def encode_categorical_variables(self, method='onehot'):
        """
        Converte variáveis categóricas para formato numérico

        Args:
            method (str): 'onehot' para One-Hot Encoding ou 'label' para Label Encoding
        """
        print("\n" + "=" * 80)
        print(f"ENCODING DE VARIÁVEIS CATEGÓRICAS (Método: {method.upper()})")
        print("=" * 80)

        if method == 'label':
            # Label Encoding - atribui número inteiro a cada categoria
            print("\nAplicando Label Encoding...")
            for col in self.categorical_cols:
                le = LabelEncoder()
                self.df_processed[col + '_encoded'] = le.fit_transform(self.df_processed[col])
                self.label_encoders[col] = le

                # Mostrar mapeamento
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                print(f"\n{col}:")
                for category, code in mapping.items():
                    print(f"  {category:20s} -> {code}")

            print("\n✓ Label Encoding concluído!")
            print(f"Colunas codificadas: {[col + '_encoded' for col in self.categorical_cols]}")

        elif method == 'onehot':
            # One-Hot Encoding - cria coluna binária para cada categoria
            print("\nAplicando One-Hot Encoding...")

            # Remover diabetes_stage se estiver nas categóricas (não usar no modelo preditivo)
            categorical_for_encoding = [col for col in self.categorical_cols if col != 'diabetes_stage']

            # Aplicar one-hot encoding
            self.df_processed = pd.get_dummies(
                self.df_processed,
                columns=categorical_for_encoding,
                prefix=categorical_for_encoding,
                drop_first=True  # Evitar multicolinearidade
            )

            print(f"\n✓ One-Hot Encoding concluído!")
            print(f"Novas dimensões: {self.df_processed.shape[0]} linhas x {self.df_processed.shape[1]} colunas")

            # Mostrar colunas criadas
            new_cols = [col for col in self.df_processed.columns if any(cat in col for cat in categorical_for_encoding)]
            print(f"\nColunas criadas ({len(new_cols)}):")
            for col in new_cols[:10]:  # Mostrar apenas primeiras 10
                print(f"  - {col}")
            if len(new_cols) > 10:
                print(f"  ... e mais {len(new_cols) - 10} colunas")

        return self


    def scale_numerical_variables(self, method='standard'):
        """
        Normaliza/Padroniza variáveis numéricas

        Args:
            method (str): 'standard' para StandardScaler ou 'minmax' para MinMaxScaler
        """
        print("\n" + "=" * 80)
        print(f"NORMALIZAÇÃO DE VARIÁVEIS NUMÉRICAS (Método: {method.upper()})")
        print("=" * 80)

        # Colunas numéricas que ainda existem no dataframe
        numerical_to_scale = [col for col in self.numerical_cols if col in self.df_processed.columns]

        if method == 'standard':
            # Padronização: média=0, desvio padrão=1
            print("\nAplicando StandardScaler (Z-score normalization)...")
            print("Fórmula: z = (x - μ) / σ")
            print("Resultado: média ≈ 0, desvio padrão ≈ 1")

            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()

        elif method == 'minmax':
            # Normalização: valores entre 0 e 1
            print("\nAplicando MinMaxScaler...")
            print("Fórmula: x_norm = (x - min) / (max - min)")
            print("Resultado: valores entre 0 e 1")

            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()

        # Aplicar transformação
        self.df_processed[numerical_to_scale] = self.scaler.fit_transform(
            self.df_processed[numerical_to_scale]
        )

        print(f"\n✓ Normalização concluída!")
        print(f"Variáveis normalizadas: {len(numerical_to_scale)}")
        print(f"\nEstatísticas após normalização:")
        print(self.df_processed[numerical_to_scale].describe().round(3))

        return self


    def correlation_analysis(self, threshold=0.7):
        """
        Realiza análise de correlação e identifica multicolinearidade

        Args:
            threshold (float): Limite para identificar correlações fortes
        """
        print("\n" + "=" * 80)
        print("ANÁLISE DE CORRELAÇÃO")
        print("=" * 80)

        # Selecionar apenas colunas numéricas
        numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns.tolist()

        # Calcular matriz de correlação
        correlation_matrix = self.df_processed[numeric_cols].corr()

        # 1. Visualização: Heatmap
        print("\n1. Gerando heatmap de correlação...")
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix,
                    annot=False,  # Não mostrar valores (muitas colunas)
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1)
        plt.title('Matriz de Correlação - Dataset Pré-processado', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("   Heatmap salvo: correlation_matrix.png")
        plt.show()

        # 2. Identificar correlações fortes
        print(f"\n2. Identificando correlações fortes (|r| >= {threshold})...")
        print("-" * 80)

        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    var1 = correlation_matrix.columns[i]
                    var2 = correlation_matrix.columns[j]
                    strong_correlations.append((var1, var2, corr_value))

        # Ordenar por correlação absoluta
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        if strong_correlations:
            print(f"\nEncontradas {len(strong_correlations)} pares com correlação forte:\n")
            for var1, var2, corr in strong_correlations:
                print(f"  {var1:40s} <-> {var2:40s}: r = {corr:6.3f}")

            print("\n⚠️  ATENÇÃO: Multicolinearidade detectada!")
            print("   Considerar:")
            print("   - Remover uma das variáveis correlacionadas")
            print("   - Usar PCA (Principal Component Analysis)")
            print("   - Usar regularização (Ridge/Lasso)")
        else:
            print(f"\nNenhuma correlação forte (|r| >= {threshold}) detectada.")
            print("✓ Boa independência entre features!")

        # 3. Correlação com variável alvo
        if self.target_col in numeric_cols:
            print(f"\n3. Correlação com variável alvo ({self.target_col}):")
            print("-" * 80)

            target_corr = correlation_matrix[self.target_col].sort_values(ascending=False)

            print(f"\nTop 15 variáveis mais correlacionadas com {self.target_col}:\n")
            for var, corr in list(target_corr.items())[1:16]:  # Pular a própria variável
                strength = "Forte" if abs(corr) >= 0.7 else "Moderada" if abs(corr) >= 0.5 else "Fraca" if abs(corr) >= 0.3 else "Muito fraca"
                direction = "positiva" if corr > 0 else "negativa"
                print(f"  {var:50s}: r = {corr:6.3f}  [{strength:12s} {direction}]")

        # 4. Calcular VIF (Variance Inflation Factor) para detectar multicolinearidade
        print("\n4. Calculando VIF (Variance Inflation Factor)...")
        print("-" * 80)
        print("VIF mede o quanto a variância de um coeficiente aumenta devido à multicolinearidade")
        print("Interpretação: VIF > 10 indica multicolinearidade severa")

        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # Selecionar features numéricas (excluir target)
        feature_cols = [col for col in numeric_cols if col != self.target_col]

        # Calcular VIF apenas para amostra (muito lento para muitas features)
        if len(feature_cols) <= 30:
            vif_data = pd.DataFrame()
            vif_data["Feature"] = feature_cols
            vif_data["VIF"] = [variance_inflation_factor(self.df_processed[feature_cols].values, i)
                               for i in range(len(feature_cols))]
            vif_data = vif_data.sort_values('VIF', ascending=False)

            print("\nTop 10 variáveis com maior VIF:")
            print(vif_data.head(10).to_string(index=False))

            high_vif = vif_data[vif_data['VIF'] > 10]
            if len(high_vif) > 0:
                print(f"\n⚠️  ATENÇÃO: {len(high_vif)} variáveis com VIF > 10 (multicolinearidade severa)")
            else:
                print("\n✓ Nenhuma variável com VIF > 10")
        else:
            print(f"\n⚠️  Dataset com {len(feature_cols)} features - VIF não calculado (computacionalmente caro)")
            print("   Sugestão: Calcular VIF após seleção de features")

        return correlation_matrix


    def create_preprocessing_pipeline(self):
        """
        Cria pipeline completo de pré-processamento usando sklearn
        """
        print("\n" + "=" * 80)
        print("CRIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
        print("=" * 80)

        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        # Definir transformações para cada tipo de variável
        categorical_for_encoding = [col for col in self.categorical_cols
                                    if col != 'diabetes_stage' and col in self.df.columns]

        # Pipeline para variáveis numéricas
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Pipeline para variáveis categóricas
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        # Combinar transformações
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, categorical_for_encoding)
            ],
            remainder='passthrough'  # Manter outras colunas
        )

        print("\n✓ Pipeline criado com sucesso!")
        print("\nComponentes do pipeline:")
        print("  1. Transformador Numérico: StandardScaler")
        print(f"     - Aplicado em {len(self.numerical_cols)} variáveis numéricas")
        print("  2. Transformador Categórico: OneHotEncoder")
        print(f"     - Aplicado em {len(categorical_for_encoding)} variáveis categóricas")

        return preprocessor


    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        """
        Prepara divisão treino/teste

        Args:
            test_size (float): Proporção de dados para teste
            random_state (int): Seed para reprodutibilidade
        """
        print("\n" + "=" * 80)
        print("PREPARANDO DIVISÃO TREINO/TESTE")
        print("=" * 80)

        # Separar features e target
        X = self.df_processed.drop(columns=[self.target_col, 'diabetes_stage'], errors='ignore')
        y = self.df_processed[self.target_col]

        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Manter proporção de classes
        )

        print(f"\n✓ Divisão realizada!")
        print(f"\nConjunto de Treino:")
        print(f"  - X_train: {X_train.shape[0]} linhas x {X_train.shape[1]} features")
        print(f"  - y_train: {y_train.shape[0]} exemplos")
        print(f"  - Distribuição de classes: {y_train.value_counts().to_dict()}")

        print(f"\nConjunto de Teste:")
        print(f"  - X_test: {X_test.shape[0]} linhas x {X_test.shape[1]} features")
        print(f"  - y_test: {y_test.shape[0]} exemplos")
        print(f"  - Distribuição de classes: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test


    def save_processed_data(self, output_path='diabetes_dataset_processed.csv'):
        """
        Salva dados processados em arquivo CSV

        Args:
            output_path (str): Caminho para salvar o arquivo
        """
        print("\n" + "=" * 80)
        print("SALVANDO DADOS PROCESSADOS")
        print("=" * 80)

        self.df_processed.to_csv(output_path, index=False)

        print(f"\n✓ Dados salvos em: {output_path}")
        print(f"Dimensões: {self.df_processed.shape[0]} linhas x {self.df_processed.shape[1]} colunas")

        return self


    def generate_preprocessing_report(self):
        """
        Gera relatório resumido do pré-processamento
        """
        print("\n" + "=" * 80)
        print("RELATÓRIO DE PRÉ-PROCESSAMENTO")
        print("=" * 80)

        print("\n1. DADOS ORIGINAIS:")
        print(f"   - Dimensões: {self.df.shape[0]} linhas x {self.df.shape[1]} colunas")
        print(f"   - Variáveis categóricas: {len(self.categorical_cols)}")
        print(f"   - Variáveis numéricas: {len(self.numerical_cols)}")

        print("\n2. DADOS PROCESSADOS:")
        print(f"   - Dimensões: {self.df_processed.shape[0]} linhas x {self.df_processed.shape[1]} colunas")
        print(f"   - Valores ausentes: {self.df_processed.isnull().sum().sum()}")
        print(f"   - Duplicatas: {self.df_processed.duplicated().sum()}")

        print("\n3. TRANSFORMAÇÕES APLICADAS:")
        print("   ✓ Remoção de duplicatas")
        print("   ✓ Tratamento de valores ausentes")
        print("   ✓ Encoding de variáveis categóricas")
        print("   ✓ Normalização de variáveis numéricas")
        print("   ✓ Análise de correlação")

        print("\n4. DATASET PRONTO PARA MODELAGEM!")
        print("   Próximos passos:")
        print("   - Seleção de features")
        print("   - Treinamento de modelos")
        print("   - Validação e avaliação")

        return self


def main():
    """
    Função principal para executar todo o pipeline de pré-processamento
    """
    print("=" * 80)
    print("PIPELINE DE PRÉ-PROCESSAMENTO - DATASET DE DIABETES")
    print("Tech Challenge IADT - Fase 1")
    print("=" * 80)

    # Inicializar preprocessador
    preprocessor = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')

    # Executar pipeline completo
    preprocessor\
        .load_data()\
        .check_data_quality()\
        .clean_data()\
        .encode_categorical_variables(method='onehot')\
        .scale_numerical_variables(method='standard')\
        .correlation_analysis(threshold=0.7)\
        .save_processed_data('diabetes_dataset_processed.csv')\
        .generate_preprocessing_report()

    # Preparar divisão treino/teste
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(
        test_size=0.2,
        random_state=42
    )

    # Criar pipeline sklearn (para uso posterior em modelos)
    sklearn_pipeline = preprocessor.create_preprocessing_pipeline()

    print("\n" + "=" * 80)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)

    return preprocessor, X_train, X_test, y_train, y_test, sklearn_pipeline


if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test, pipeline = main()
