# Módulo de Pré-processamento de Dados - Dataset de Diabetes

Este módulo contém o pipeline completo de pré-processamento para o dataset de diabetes do Tech Challenge IADT - Fase 1.

## Arquivos

- **[preprocessing.py](preprocessing.py)**: Módulo principal com a classe `DiabetesDataPreprocessor`
- **[exemplo_uso_preprocessing.py](exemplo_uso_preprocessing.py)**: Exemplos de uso do módulo
- **README_PREPROCESSING.md**: Esta documentação

## Funcionalidades

### 1. Limpeza de Dados
- Detecção e tratamento de valores ausentes
- Remoção de duplicatas
- Identificação de valores inconsistentes
- Validação de tipos de dados

### 2. Transformação de Variáveis Categóricas
- **Label Encoding**: Converte categorias em números inteiros (0, 1, 2, ...)
- **One-Hot Encoding**: Cria colunas binárias para cada categoria (recomendado)

### 3. Normalização de Variáveis Numéricas
- **StandardScaler**: Padronização (média=0, desvio padrão=1)
- **MinMaxScaler**: Normalização (valores entre 0 e 1)

### 4. Análise de Correlação
- Matriz de correlação com visualização (heatmap)
- Identificação de multicolinearidade
- Cálculo de VIF (Variance Inflation Factor)
- Correlação com variável alvo

### 5. Pipeline Sklearn
- Criação de pipeline compatível com scikit-learn
- Integração fácil com modelos de machine learning

### 6. Divisão Treino/Teste
- Split estratificado (mantém proporção de classes)
- Configurável (test_size, random_state)

## Uso Básico

```python
from preprocessing import DiabetesDataPreprocessor

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

# Preparar dados para modelagem
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()
```

## Execução Rápida

### Pipeline Completo

```bash
python preprocessing.py
```

Este comando executa todo o pipeline e gera:
- Dataset processado: `diabetes_dataset_processed.csv`
- Heatmap de correlação: `correlation_matrix.png`
- Relatório completo no console

### Exemplos de Uso

```bash
python exemplo_uso_preprocessing.py
```

Demonstra diferentes cenários de uso do módulo.

## Estrutura da Classe DiabetesDataPreprocessor

### Métodos Principais

#### `load_data()`
Carrega o dataset do arquivo CSV.

```python
preprocessor.load_data()
```

#### `check_data_quality()`
Realiza análise de qualidade dos dados:
- Valores ausentes
- Duplicatas
- Tipos de dados
- Valores inconsistentes
- Distribuição da variável alvo

```python
preprocessor.check_data_quality()
```

#### `clean_data()`
Realiza limpeza dos dados:
- Remove duplicatas
- Preenche valores ausentes (mediana para numéricas, moda para categóricas)
- Valida tipos de dados

```python
preprocessor.clean_data()
```

#### `encode_categorical_variables(method='onehot')`
Converte variáveis categóricas para formato numérico.

**Parâmetros:**
- `method`: `'onehot'` ou `'label'`

```python
# One-Hot Encoding (recomendado)
preprocessor.encode_categorical_variables(method='onehot')

# Label Encoding
preprocessor.encode_categorical_variables(method='label')
```

#### `scale_numerical_variables(method='standard')`
Normaliza variáveis numéricas.

**Parâmetros:**
- `method`: `'standard'` (StandardScaler) ou `'minmax'` (MinMaxScaler)

```python
# Padronização (média=0, std=1)
preprocessor.scale_numerical_variables(method='standard')

# Normalização (valores entre 0 e 1)
preprocessor.scale_numerical_variables(method='minmax')
```

#### `correlation_analysis(threshold=0.7)`
Realiza análise de correlação.

**Parâmetros:**
- `threshold`: Limite para identificar correlações fortes (padrão: 0.7)

```python
correlation_matrix = preprocessor.correlation_analysis(threshold=0.7)
```

**Saídas:**
- Heatmap de correlação salvo em `correlation_matrix.png`
- Lista de correlações fortes
- VIF para detectar multicolinearidade
- Top features correlacionadas com variável alvo

#### `prepare_train_test_split(test_size=0.2, random_state=42)`
Divide dados em conjuntos de treino e teste.

**Parâmetros:**
- `test_size`: Proporção de dados para teste (padrão: 0.2 = 20%)
- `random_state`: Seed para reprodutibilidade (padrão: 42)

```python
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(
    test_size=0.2,
    random_state=42
)
```

#### `create_preprocessing_pipeline()`
Cria pipeline do scikit-learn para uso em modelos.

```python
sklearn_pipeline = preprocessor.create_preprocessing_pipeline()

# Usar em um modelo
from sklearn.ensemble import RandomForestClassifier

model = Pipeline([
    ('preprocessor', sklearn_pipeline),
    ('classifier', RandomForestClassifier())
])
```

#### `save_processed_data(output_path='diabetes_dataset_processed.csv')`
Salva dados processados em arquivo CSV.

```python
preprocessor.save_processed_data('diabetes_processed.csv')
```

#### `generate_preprocessing_report()`
Gera relatório resumido do pré-processamento.

```python
preprocessor.generate_preprocessing_report()
```

## Variáveis do Dataset

### Variáveis Categóricas (7)
- `gender`: Gênero (Male, Female, Other)
- `ethnicity`: Etnia (Asian, White, Hispanic, Black, Other)
- `education_level`: Nível educacional (Highschool, Graduate, Postgraduate, No formal)
- `income_level`: Nível de renda (Low, Lower-Middle, Middle, Upper-Middle, High)
- `employment_status`: Status de emprego (Employed, Unemployed, Retired, Student)
- `smoking_status`: Status de fumante (Never, Former, Current)
- `diabetes_stage`: Estágio do diabetes (Type 1, Type 2, Pre-Diabetes, Gestational, No Diabetes)

### Variáveis Numéricas (23)
- `age`: Idade
- `bmi`: Índice de Massa Corporal
- `glucose_fasting`: Glicose em jejum
- `glucose_postprandial`: Glicose pós-prandial
- `hba1c`: Hemoglobina glicada
- `insulin_level`: Nível de insulina
- `diabetes_risk_score`: Score de risco de diabetes
- `cholesterol_total`: Colesterol total
- `hdl_cholesterol`: HDL (colesterol "bom")
- `ldl_cholesterol`: LDL (colesterol "ruim")
- `triglycerides`: Triglicerídeos
- `systolic_bp`: Pressão sistólica
- `diastolic_bp`: Pressão diastólica
- `physical_activity_minutes_per_week`: Atividade física (min/semana)
- `sleep_hours_per_day`: Horas de sono por dia
- `diet_score`: Score de dieta (0-10)
- `alcohol_consumption_per_week`: Consumo de álcool (doses/semana)
- `screen_time_hours_per_day`: Tempo de tela (horas/dia)
- `family_history_diabetes`: Histórico familiar de diabetes (0/1)
- `hypertension_history`: Histórico de hipertensão (0/1)
- `cardiovascular_history`: Histórico cardiovascular (0/1)
- `waist_to_hip_ratio`: Relação cintura/quadril
- `heart_rate`: Frequência cardíaca

### Variável Alvo
- `diagnosed_diabetes`: Diagnóstico de diabetes (0 = Não, 1 = Sim)

## Exemplos de Uso

### Exemplo 1: Pipeline Completo

```python
from preprocessing import DiabetesDataPreprocessor

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
```

### Exemplo 2: Preparar Dados para Modelagem

```python
from preprocessing import DiabetesDataPreprocessor

preprocessor = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')

preprocessor\
    .load_data()\
    .clean_data()\
    .encode_categorical_variables(method='onehot')\
    .scale_numerical_variables(method='standard')

# Dividir dados
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(
    test_size=0.2,
    random_state=42
)

# Treinar modelo
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Avaliar
accuracy = model.score(X_test, y_test)
print(f"Acurácia: {accuracy:.2%}")
```

### Exemplo 3: Apenas Análise Exploratória

```python
from preprocessing import DiabetesDataPreprocessor

preprocessor = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')

# Apenas análise de qualidade
preprocessor.load_data().check_data_quality()

# Apenas análise de correlação
preprocessor.load_data().correlation_analysis(threshold=0.5)
```

### Exemplo 4: Diferentes Métodos de Encoding

```python
from preprocessing import DiabetesDataPreprocessor

# One-Hot Encoding (recomendado para a maioria dos casos)
prep_onehot = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')
prep_onehot.load_data().clean_data().encode_categorical_variables(method='onehot')

# Label Encoding (útil para árvores de decisão)
prep_label = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')
prep_label.load_data().clean_data().encode_categorical_variables(method='label')
```

### Exemplo 5: Diferentes Métodos de Scaling

```python
from preprocessing import DiabetesDataPreprocessor

# StandardScaler (recomendado para a maioria dos casos)
prep_standard = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')
prep_standard.load_data().clean_data().scale_numerical_variables(method='standard')

# MinMaxScaler (útil para redes neurais)
prep_minmax = DiabetesDataPreprocessor(data_path='diabetes_dataset.csv')
prep_minmax.load_data().clean_data().scale_numerical_variables(method='minmax')
```

## Análise de Correlação

### Interpretação dos Valores de Correlação

- **|r| >= 0.7**: Correlação forte
- **0.5 <= |r| < 0.7**: Correlação moderada
- **0.3 <= |r| < 0.5**: Correlação fraca
- **|r| < 0.3**: Correlação muito fraca ou inexistente

### VIF (Variance Inflation Factor)

Usado para detectar multicolinearidade:
- **VIF < 5**: Sem multicolinearidade
- **5 <= VIF < 10**: Multicolinearidade moderada
- **VIF >= 10**: Multicolinearidade severa (remover variável)

## Recomendações

### Para Modelos Lineares (Regressão Logística, SVM)
- Use **One-Hot Encoding** para categóricas
- Use **StandardScaler** para numéricas
- Remova variáveis com VIF > 10
- Considere regularização (Ridge/Lasso)

### Para Árvores de Decisão (Random Forest, XGBoost)
- Use **Label Encoding** ou **One-Hot Encoding**
- **Não** é necessário normalizar
- Árvores são robustas à multicolinearidade

### Para Redes Neurais
- Use **One-Hot Encoding** para categóricas
- Use **MinMaxScaler** para numéricas (valores entre 0 e 1)
- Remova features redundantes

## Troubleshooting

### Erro: "FileNotFoundError"
Certifique-se de que o arquivo `diabetes_dataset.csv` está no mesmo diretório.

### Erro: "Memory Error"
Dataset muito grande. Considere:
- Processar em chunks
- Reduzir número de features
- Usar servidor com mais memória

### Warning: "SettingWithCopyWarning"
Ignore - o código usa `.copy()` apropriadamente.

### VIF não calculado
Dataset com muitas features (> 30). VIF é computacionalmente caro.
Sugestão: Calcule VIF após seleção inicial de features.

## Próximos Passos

Após o pré-processamento, você pode:

1. **Seleção de Features**
   - SelectKBest
   - Feature Importance (Random Forest)
   - PCA (Principal Component Analysis)

2. **Treinamento de Modelos**
   - Regressão Logística
   - Random Forest
   - XGBoost
   - SVM
   - Redes Neurais

3. **Validação e Avaliação**
   - Cross-validation
   - Métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Matriz de confusão

4. **Otimização**
   - Grid Search / Random Search
   - Bayesian Optimization
   - Ensemble methods

## Contato

Tech Challenge IADT - Fase 1
- Hiago Marques Rubio
- Mylena Ferreira Lacerda
