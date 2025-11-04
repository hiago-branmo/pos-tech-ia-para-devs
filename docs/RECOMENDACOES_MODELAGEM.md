# Recomendações de Modelagem - Dataset de Diabetes

## Problema

**Tipo:** Classificação Binária Supervisionada
**Variável Alvo:** `diagnosed_diabetes` (0 = Sem diabetes, 1 = Com diabetes)
**Dataset:** ~100.000 exemplos, 31 variáveis (7 categóricas + 23 numéricas + 1 target)

---

## 1. Features: O que Usar e o que Remover

### ❌ REMOVER OBRIGATORIAMENTE

#### 1.1 `diabetes_stage`
**Motivo:** DATA LEAKAGE (vazamento de dados)
- Esta coluna indica o estágio do diabetes (Type 1, Type 2, Pre-Diabetes, Gestational, No Diabetes)
- É uma consequência DIRETA do diagnóstico
- Usar esta variável é como "trapacear" - o modelo terá 100% de acurácia mas não funcionará em produção
- **Analogia:** É como prever se alguém passou no vestibular usando a informação "está matriculado na universidade"

#### 1.2 `diabetes_risk_score`
**Motivo:** REDUNDÂNCIA / MULTICOLINEARIDADE
- Este score provavelmente é calculado a partir de outras variáveis do dataset
- Usar o score + suas componentes = multicolinearidade severa
- **Opção A:** Remover completamente (recomendado)
- **Opção B:** Usar APENAS o score e remover algumas variáveis clínicas

### ⚠️ CONSIDERAR REMOVER (Multicolinearidade)

#### 1.3 Variáveis Glicêmicas (escolher 1 ou 2)
- `glucose_fasting` (glicose em jejum)
- `glucose_postprandial` (glicose pós-refeição)
- `hba1c` (hemoglobina glicada - média de 2-3 meses)

**Recomendação:** Manter apenas `hba1c` (marcador gold standard)
- HbA1c é o critério diagnóstico oficial (≥ 6.5% = diabetes)
- Já reflete média das glicemias ao longo do tempo
- Remover as duas glicemias pontuais reduz multicolinearidade

**OU:** Criar feature composta
```python
df['glucose_avg'] = (df['glucose_fasting'] + df['glucose_postprandial']) / 2
# E remover glucose_fasting e glucose_postprandial
```

#### 1.4 Colesterol Total vs HDL + LDL
- `cholesterol_total` é aproximadamente = HDL + LDL + triglicerídeos/5

**Recomendação:** Remover `cholesterol_total` e manter:
- `hdl_cholesterol` (colesterol "bom")
- `ldl_cholesterol` (colesterol "ruim")
- `triglycerides`

### ✅ FEATURES IMPORTANTES A MANTER

#### Marcadores Clínicos Diretos
- ✅ `hba1c` - Gold standard para diabetes
- ✅ `insulin_level` - Resistência insulínica
- ✅ `bmi` - Obesidade (fator de risco principal)
- ✅ `waist_to_hip_ratio` - Obesidade central

#### Fatores de Risco Cardiovasculares
- ✅ `systolic_bp` e `diastolic_bp` - Hipertensão
- ✅ `hdl_cholesterol`, `ldl_cholesterol`, `triglycerides`

#### Estilo de Vida
- ✅ `physical_activity_minutes_per_week` - Sedentarismo
- ✅ `diet_score` - Qualidade alimentar
- ✅ `sleep_hours_per_day`
- ✅ `alcohol_consumption_per_week`
- ✅ `screen_time_hours_per_day`
- ✅ `smoking_status` (categórica)

#### Demográficas e Histórico
- ✅ `age` - Idade
- ✅ `gender` - Gênero (categórica)
- ✅ `family_history_diabetes` - Genética
- ✅ `hypertension_history` - Comorbidade
- ✅ `cardiovascular_history` - Comorbidade

#### Socioeconômicas (opcional - testar importância)
- ⚠️ `ethnicity`
- ⚠️ `education_level`
- ⚠️ `income_level`
- ⚠️ `employment_status`

**Nota:** Variáveis socioeconômicas podem ter pouco poder preditivo mas são importantes para análise de equidade/fairness do modelo.

---

## 2. Modelos Recomendados

### 🥇 Tier 1: ALTAMENTE RECOMENDADOS

#### 2.1 **Random Forest Classifier**
**Por que usar:**
- ✅ Funciona muito bem com dados tabulares
- ✅ Robusto a outliers e missing values
- ✅ Não requer normalização
- ✅ Feature importance automática
- ✅ Bom com dados desbalanceados (ajustar class_weight)
- ✅ Baixo risco de overfitting (se n_estimators alto)

**Configuração sugerida:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  # Para dados desbalanceados
    random_state=42,
    n_jobs=-1  # Usar todos os cores
)
```

**Quando usar:**
- Primeira linha de análise
- Quando interpretabilidade moderada é suficiente
- Quando há suspeita de interações não-lineares

---

#### 2.2 **XGBoost / LightGBM / CatBoost**
**Por que usar:**
- ✅ ESTADO DA ARTE para dados tabulares
- ✅ Melhor performance que Random Forest (geralmente)
- ✅ Controle fino de overfitting (early stopping, regularização)
- ✅ Rápido e eficiente
- ✅ Feature importance detalhada

**XGBoost - Configuração sugerida:**
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  # Ajustar se desbalanceado
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50  # Parar se não melhorar
)
```

**LightGBM - Mais rápido:**
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    num_leaves=31,
    class_weight='balanced',
    random_state=42
)
```

**Quando usar:**
- Quando busca melhor performance possível
- Competições (Kaggle-style)
- Produção (deploy em sistemas reais)

---

#### 2.3 **Regressão Logística (com Regularização)**
**Por que usar:**
- ✅ MELHOR INTERPRETABILIDADE (coeficientes = efeitos)
- ✅ Rápido para treinar
- ✅ Baseline sólido
- ✅ Probabilidades calibradas
- ✅ Bom para entender relações lineares

**Configuração sugerida:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty='l2',  # Ridge (ou 'l1' para Lasso)
    C=1.0,  # Inverso da força de regularização
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)
```

**Quando usar:**
- Quando INTERPRETABILIDADE é crítica (medicina, regulatório)
- Para entender relações causais
- Como baseline comparativo
- Quando modelo precisa ser explicável a não-técnicos

**⚠️ Requer:**
- Normalização (StandardScaler) - já feito no preprocessing
- Baixa multicolinearidade (remover features correlacionadas)

---

### 🥈 Tier 2: BONS COMPLEMENTARES

#### 2.4 **Support Vector Machine (SVM)**
**Por que usar:**
- ✅ Bom com dados de alta dimensão
- ✅ Funciona bem com kernel RBF (não-linearidade)

**Configuração sugerida:**
```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,  # Para obter probabilidades
    random_state=42
)
```

**Quando usar:**
- Dataset médio (< 50k exemplos) - SVM é lento
- Quando há separação não-linear clara
- Como modelo complementar em ensemble

**⚠️ Limitações:**
- Lento com datasets grandes (100k pode ser problema)
- Menos interpretável que Logistic Regression

---

#### 2.5 **K-Nearest Neighbors (KNN)**
**Por que usar:**
- ✅ Simples e intuitivo
- ✅ Não assume distribuição dos dados

**Configuração sugerida:**
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=11,  # Testar 5, 11, 21
    weights='distance',  # Vizinhos mais próximos têm mais peso
    metric='minkowski',
    n_jobs=-1
)
```

**Quando usar:**
- Baseline simples
- Análise exploratória

**⚠️ Limitações:**
- Lento para predição (100k exemplos)
- Sensível a features irrelevantes
- "Curse of dimensionality"

---

#### 2.6 **Redes Neurais (MLP)**
**Por que usar:**
- ✅ Captura interações complexas não-lineares
- ✅ Flexível

**Configuração sugerida:**
```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(100, 50, 25),  # 3 camadas ocultas
    activation='relu',
    solver='adam',
    alpha=0.0001,  # Regularização L2
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
```

**Quando usar:**
- Quando há muitas interações não-lineares
- Dataset grande (> 50k exemplos)
- Quando tree-based models não funcionam bem

**⚠️ Limitações:**
- Difícil de interpretar (caixa-preta)
- Requer tuning cuidadoso
- Pode overfittar facilmente

---

### 🥉 Tier 3: ANÁLISES COMPLEMENTARES

#### 2.7 **Naive Bayes**
**Por que usar:**
- ✅ Extremamente rápido
- ✅ Funciona com pouco dado

**Quando usar:**
- Baseline ultra-rápido
- Quando dados são realmente "naive" (features independentes)

**⚠️ Limitações:**
- Assume independência entre features (raramente verdade)
- Performance geralmente inferior

---

#### 2.8 **Decision Tree (Árvore única)**
**Por que usar:**
- ✅ MÁXIMA INTERPRETABILIDADE (pode visualizar árvore)
- ✅ Não requer normalização

**Quando usar:**
- Análise exploratória
- Entender estrutura de decisão
- Baseline (Random Forest sempre melhor)

**⚠️ Limitações:**
- Alto risco de overfitting
- Instável (pequenas mudanças nos dados = árvore diferente)

---

## 3. Estratégia de Modelagem Recomendada

### Pipeline Sugerido

```
1. BASELINE SIMPLES
   └─ Logistic Regression (interpretável, rápido)

2. TREE-BASED MODELS
   └─ Random Forest (robusto, feature importance)
   └─ XGBoost/LightGBM (melhor performance)

3. ENSEMBLE (combinar modelos)
   └─ Voting Classifier (RF + XGB + LR)
   └─ Stacking (usar predições como features)

4. ANÁLISE DE RESULTADOS
   └─ Comparar métricas
   └─ Análise de erros (FP, FN)
   └─ Feature importance
   └─ SHAP values (explicabilidade)
```

### Métricas de Avaliação

**Para dataset balanceado (~50/50):**
- ✅ **Accuracy** (acurácia geral)
- ✅ **ROC-AUC** (área sob curva ROC)
- ✅ **F1-Score** (média harmônica de Precision e Recall)

**Se dataset desbalanceado:**
- ✅ **Precision** (de positivos preditos, quantos são reais?)
- ✅ **Recall** (de positivos reais, quantos detectamos?)
- ✅ **PR-AUC** (área sob curva Precision-Recall)
- ✅ **Confusion Matrix** (analisar FP e FN)

**Contexto Médico (diabetes):**
- 🏥 **Recall é mais importante** (não queremos perder casos de diabetes - Falso Negativo é pior)
- ⚠️ Mas Precision também importa (muitos Falsos Positivos = exames desnecessários, ansiedade)

---

## 4. Feature Selection (Seleção de Features)

### Métodos Recomendados

#### 4.1 Feature Importance (Random Forest / XGBoost)
```python
# Treinar Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Obter importâncias
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Selecionar top N features
top_features = importances.head(20)['feature'].tolist()
```

#### 4.2 SelectKBest (Estatístico)
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

#### 4.3 Recursive Feature Elimination (RFE)
```python
from sklearn.feature_selection import RFE

estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=20, step=1)
selector.fit(X_train, y_train)
```

#### 4.4 Remoção Manual Baseada em Correlação
```python
# Já feito no preprocessing - remover features com |r| > 0.9
```

---

## 5. Lista Final de Features Recomendadas

### Conjunto Mínimo (15-20 features) - RECOMENDADO PARA COMEÇAR

```python
features_essenciais = [
    # Marcadores clínicos diretos
    'hba1c',                    # Gold standard diabetes
    'insulin_level',            # Resistência insulínica
    'bmi',                      # Obesidade
    'waist_to_hip_ratio',       # Obesidade central

    # Cardiovascular
    'systolic_bp',              # Hipertensão
    'diastolic_bp',             # Hipertensão
    'hdl_cholesterol',          # Colesterol bom
    'ldl_cholesterol',          # Colesterol ruim
    'triglycerides',            # Lipídios

    # Estilo de vida
    'physical_activity_minutes_per_week',  # Sedentarismo
    'diet_score',               # Alimentação
    'age',                      # Idade

    # Histórico
    'family_history_diabetes',  # Genética
    'hypertension_history',     # Comorbidade
    'cardiovascular_history',   # Comorbidade

    # Categóricas (após one-hot encoding)
    'gender_*',                 # Gênero
    'smoking_status_*'          # Tabagismo
]
```

### Conjunto Completo (todas menos as removidas)

```python
features_completas = [
    # Todas as numéricas EXCETO:
    # - diabetes_stage (data leakage)
    # - diabetes_risk_score (redundante)
    # - glucose_fasting (redundante com hba1c)
    # - glucose_postprandial (redundante com hba1c)
    # - cholesterol_total (redundante com HDL+LDL)

    # Todas as categóricas (one-hot encoded) EXCETO:
    # - diabetes_stage
]
```

---

## 6. Código Exemplo - Pipeline Completo

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

# Carregar dados pré-processados
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Remover features problemáticas
features_to_drop = [
    'diabetes_risk_score',
    'glucose_fasting',
    'glucose_postprandial',
    'cholesterol_total'
]
X_train = X_train.drop(columns=features_to_drop, errors='ignore')
X_test = X_test.drop(columns=features_to_drop, errors='ignore')

# Remover colunas de diabetes_stage se existirem (one-hot encoded)
diabetes_stage_cols = [col for col in X_train.columns if 'diabetes_stage' in col.lower()]
X_train = X_train.drop(columns=diabetes_stage_cols, errors='ignore')
X_test = X_test.drop(columns=diabetes_stage_cols, errors='ignore')

print(f"Features finais: {X_train.shape[1]}")

# ========================================
# MODELO 1: Logistic Regression (Baseline)
# ========================================
print("\n" + "="*80)
print("MODELO 1: LOGISTIC REGRESSION")
print("="*80)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")

# ========================================
# MODELO 2: Random Forest
# ========================================
print("\n" + "="*80)
print("MODELO 2: RANDOM FOREST")
print("="*80)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")

# ========================================
# MODELO 3: XGBoost
# ========================================
print("\n" + "="*80)
print("MODELO 3: XGBOOST")
print("="*80)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_xgb):.4f}")

# ========================================
# COMPARAÇÃO FINAL
# ========================================
print("\n" + "="*80)
print("COMPARAÇÃO DOS MODELOS")
print("="*80)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_proba_lr),
        roc_auc_score(y_test, y_proba_rf),
        roc_auc_score(y_test, y_proba_xgb)
    ]
})

print(results.to_string(index=False))
```

---

## 7. Resumo Executivo

### Features a USAR (após one-hot encoding):
- ✅ Todas as variáveis numéricas EXCETO: `diabetes_stage`, `diabetes_risk_score`, `glucose_fasting`, `glucose_postprandial`, `cholesterol_total`
- ✅ Todas as variáveis categóricas EXCETO: `diabetes_stage`

### Modelos Recomendados (ordem de prioridade):
1. 🥇 **XGBoost/LightGBM** - Melhor performance
2. 🥇 **Random Forest** - Robusto e confiável
3. 🥇 **Logistic Regression** - Interpretável (baseline)
4. 🥈 **SVM** - Complementar
5. 🥈 **Neural Network** - Se houver tempo

### Métricas Principais:
- 📊 **ROC-AUC** (métrica principal)
- 📊 **F1-Score** (balanceamento Precision/Recall)
- 📊 **Recall** (importante em contexto médico)
- 📊 **Confusion Matrix** (análise de erros)

### Next Steps:
1. Executar preprocessing.ipynb
2. Criar notebook de modelagem com os 3 modelos principais
3. Comparar resultados
4. Analisar feature importance
5. Tunning de hiperparâmetros (Grid Search)
6. Ensemble (combinar modelos)
