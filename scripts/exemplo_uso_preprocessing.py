"""
Exemplo de Uso do Módulo de Pré-processamento
Tech Challenge IADT - Fase 1

Este arquivo demonstra como usar a classe DiabetesDataPreprocessor
para diferentes cenários de pré-processamento.
"""

from preprocessing import DiabetesDataPreprocessor


# ============================================================================
# EXEMPLO 1: Pipeline Completo (uso básico)
# ============================================================================
print("\n" + "="*80)
print("EXEMPLO 1: PIPELINE COMPLETO")
print("="*80 + "\n")

# Inicializar e executar pipeline completo
preprocessor = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')

preprocessor\
    .load_data()\
    .check_data_quality()\
    .clean_data()\
    .encode_categorical_variables(method='onehot')\
    .scale_numerical_variables(method='standard')\
    .correlation_analysis(threshold=0.7)\
    .save_processed_data('diabetes_processed_full.csv')\
    .generate_preprocessing_report()


# ============================================================================
# EXEMPLO 2: Apenas Análise de Qualidade (sem transformações)
# ============================================================================
print("\n" + "="*80)
print("EXEMPLO 2: APENAS ANÁLISE DE QUALIDADE")
print("="*80 + "\n")

preprocessor_check = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')
preprocessor_check.load_data().check_data_quality()


# ============================================================================
# EXEMPLO 3: Pipeline com Label Encoding (ao invés de One-Hot)
# ============================================================================
print("\n" + "="*80)
print("EXEMPLO 3: PIPELINE COM LABEL ENCODING")
print("="*80 + "\n")

preprocessor_label = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')

preprocessor_label\
    .load_data()\
    .clean_data()\
    .encode_categorical_variables(method='label')\
    .scale_numerical_variables(method='standard')\
    .save_processed_data('diabetes_processed_label.csv')


# ============================================================================
# EXEMPLO 4: Pipeline com MinMax Scaling (ao invés de Standard)
# ============================================================================
print("\n" + "="*80)
print("EXEMPLO 4: PIPELINE COM MINMAX SCALING")
print("="*80 + "\n")

preprocessor_minmax = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')

preprocessor_minmax\
    .load_data()\
    .clean_data()\
    .encode_categorical_variables(method='onehot')\
    .scale_numerical_variables(method='minmax')\
    .save_processed_data('diabetes_processed_minmax.csv')


# ============================================================================
# EXEMPLO 5: Preparar dados para Modelagem
# ============================================================================
print("\n" + "="*80)
print("EXEMPLO 5: PREPARAR DADOS PARA MODELAGEM")
print("="*80 + "\n")

preprocessor_ml = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')

preprocessor_ml\
    .load_data()\
    .clean_data()\
    .encode_categorical_variables(method='onehot')\
    .scale_numerical_variables(method='standard')

# Dividir em treino/teste
X_train, X_test, y_train, y_test = preprocessor_ml.prepare_train_test_split(
    test_size=0.2,
    random_state=42
)

# Criar pipeline sklearn
sklearn_pipeline = preprocessor_ml.create_preprocessing_pipeline()

print("\n✓ Dados prontos para treinar modelos!")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape: {y_test.shape}")


# ============================================================================
# EXEMPLO 6: Apenas Análise de Correlação
# ============================================================================
print("\n" + "="*80)
print("EXEMPLO 6: APENAS ANÁLISE DE CORRELAÇÃO")
print("="*80 + "\n")

preprocessor_corr = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')

preprocessor_corr.load_data()

# Análise de correlação com limiar customizado
correlation_matrix = preprocessor_corr.correlation_analysis(threshold=0.5)

print("\nMatriz de correlação armazenada para uso posterior!")


# ============================================================================
# EXEMPLO 7: Acesso aos Dados Processados
# ============================================================================
print("\n" + "="*80)
print("EXEMPLO 7: ACESSO AOS DADOS PROCESSADOS")
print("="*80 + "\n")

preprocessor_data = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')

preprocessor_data\
    .load_data()\
    .clean_data()\
    .encode_categorical_variables(method='onehot')\
    .scale_numerical_variables(method='standard')

# Acessar dataframe processado
df_processed = preprocessor_data.df_processed

print(f"DataFrame processado disponível:")
print(f"  - Shape: {df_processed.shape}")
print(f"  - Colunas: {df_processed.columns.tolist()[:10]}... (primeiras 10)")
print(f"\nPrimeiras 5 linhas:")
print(df_processed.head())


print("\n" + "="*80)
print("TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
print("="*80 + "\n")
