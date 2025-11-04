# Sistema de Suporte ao Diagnóstico de Diabetes

## Tech Challenge FIAP - Fase 1

### Autores

Hiago Marques Rubio
Mylena Ferreira Lacerda

---

## Sobre o Projeto

Este projeto desenvolve um sistema inteligente de suporte ao diagnóstico médico usando Machine Learning. O foco é criar uma ferramenta que ajude profissionais de saúde na análise inicial de exames e dados clínicos de pacientes, especificamente para detecção de diabetes.

O sistema analisa dados médicos estruturados e fornece predições acompanhadas de explicações detalhadas, permitindo que médicos entendam quais fatores influenciaram cada diagnóstico.

### O Problema

Hospitais enfrentam volume crescente de pacientes e exames. A triagem manual é demorada e propensa a erros. Este sistema busca acelerar o processo de análise inicial mantendo transparência e interpretabilidade das decisões.

### Importante

Este é um sistema de SUPORTE à decisão. O médico sempre tem a palavra final no diagnóstico. A ferramenta auxilia, mas não substitui o julgamento clínico profissional.

---

## Dataset

**Fonte:** Diabetes Health Indicators Dataset (Kaggle)

O dataset contém aproximadamente 100.000 registros de pacientes com 31 variáveis incluindo dados demográficos, histórico médico, exames laboratoriais e indicadores de saúde.

**Variável alvo:** diagnosed_diabetes (0 = sem diabetes, 1 = com diabetes)

**Principais features:**
HbA1c, glicose em jejum, glicose pós-prandial, IMC, idade, pressão arterial, colesterol, histórico familiar, entre outras.

---

## Estrutura do Projeto

```
fase-1/
│
├── notebooks/
│   ├── analise-exploratoria.ipynb    # EDA completa com estatísticas e visualizações
│   ├── preprocessing.ipynb            # Pipeline de limpeza e transformação
│   ├── modelagem.ipynb                # Definição e configuração dos modelos
│   └── treinamento.ipynb              # Treinamento, avaliação e SHAP
│
├── data/
│   ├── diabetes_dataset.csv           # Dataset original
│   ├── X_train.csv, X_test.csv        # Features processadas
│   └── y_train.csv, y_test.csv        # Labels
│
├── models/
│   ├── logistic_regression_model.pkl  # Modelo baseline
│   ├── random_forest_model.pkl        # Modelo ensemble
│   └── xgboost_model.pkl              # Modelo vencedor
│
├── scripts/
│   ├── preprocessing.py               # Classe reutilizável de pré-processamento
│   └── exemplo_uso_preprocessing.py   # Exemplos de uso
│
└── docs/
    ├── GUIA_INTERPRETACAO_RESULTADOS.md
    ├── RECOMENDACOES_MODELAGEM.md
    └── README_PREPROCESSING.md
```

---

## Metodologia

### 1. Análise Exploratória de Dados

Análise estatística completa de 23 variáveis numéricas com histogramas, boxplots, quartis e identificação de outliers. Estudo de correlações entre variáveis e discussão sobre causalidade vs correlação.

**Notebook:** notebooks/analise-exploratoria.ipynb

### 2. Pré-processamento

Pipeline completo incluindo tratamento de valores ausentes, remoção de duplicatas, One-Hot Encoding para variáveis categóricas e StandardScaler para normalização. Análise de multicolinearidade com VIF e matriz de correlação.

Remoção de features com data leakage (diabetes_stage) e features redundantes (diabetes_risk_score, glucose_fasting, glucose_postprandial, cholesterol_total).

Split estratificado 80/20 mantendo proporção de classes.

**Notebook:** notebooks/preprocessing.ipynb

### 3. Modelagem

Três modelos foram treinados e avaliados:

**Logistic Regression**
Modelo baseline interpretável. Coeficientes indicam diretamente o impacto de cada feature. Ideal para contexto médico pela transparência.

**Random Forest**
Ensemble de árvores. Captura relações não-lineares e é robusto a overfitting. Precision de 99.98% no conjunto de teste.

**XGBoost**
Modelo vencedor com ROC-AUC de 0.9413. Melhor balanço entre performance e generalização. Usado para análise SHAP detalhada.

**Notebooks:** notebooks/modelagem.ipynb e notebooks/treinamento.ipynb

### 4. Avaliação

Métricas completas para classificação binária: Accuracy, Precision, Recall, F1-Score e ROC-AUC. Análise de Confusion Matrix com foco em Falsos Negativos (mais críticos em contexto médico).

Validação Cruzada (5-fold Stratified) confirma robustez dos modelos. Diferença entre CV e Test Set menor que 2% indica boa generalização.

### 5. Explicabilidade (SHAP)

Análise SHAP (SHapley Additive exPlanations) para todos os modelos. SHAP explica POR QUE o modelo fez cada predição, essencial para aceitação médica.

**Summary Plots:** Mostram importância global de features
**Waterfall Plots:** Explicam predições individuais passo a passo
**Dependence Plots:** Revelam relações não-lineares

Principais insights: HbA1c é consistentemente a feature mais importante (alinhado com conhecimento médico), seguido por glicose, idade e IMC.

---

## Resultados

### Comparação de Modelos

| Modelo              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.8855   | 0.9290    | 0.8762 | 0.9018   | 0.9336  |
| Random Forest       | 0.9103   | 0.9998    | 0.8506 | 0.9192   | 0.9405  |
| XGBoost             | 0.9101   | 0.9988    | 0.8511 | 0.9191   | 0.9413  |

### Modelo Vencedor: XGBoost

ROC-AUC de 0.9413 indica excelente capacidade de discriminação. Accuracy de 91% com Precision próxima de 100% minimiza falsos positivos.

### Contexto Médico

Recall de 85% significa que 15% dos casos de diabetes podem não ser detectados (Falsos Negativos). Em aplicação real, recomenda-se ajustar threshold para priorizar Recall, mesmo aumentando falsos positivos, pois exames adicionais têm menor custo que diabetes não tratada.

---

## Como Executar

### Requisitos

Python 3.9 ou superior

Principais bibliotecas: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap

### Instalação

```bash
# Clonar repositório
git clone [URL_DO_REPOSITORIO]
cd fase-1

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Instalar dependências
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter shap
```

### Nota para usuários macOS

XGBoost requer OpenMP. Se encontrar erro de biblioteca, execute:

```bash
brew install libomp
```

Depois reinicie o kernel do Jupyter.

### Executar Notebooks

```bash
# Iniciar Jupyter
jupyter notebook

# Executar na ordem:
# 1. notebooks/analise-exploratoria.ipynb
# 2. notebooks/preprocessing.ipynb
# 3. notebooks/modelagem.ipynb
# 4. notebooks/treinamento.ipynb
```

Cada notebook gera os arquivos necessários para o próximo. Os dados processados ficam em data/ e os modelos treinados em models/.

---

## Aplicação Prática

### Fluxo Recomendado

1. Sistema recebe dados do paciente (exames laboratoriais, dados demográficos)
2. Pré-processamento aplica mesmas transformações do treino
3. Modelo faz predição e gera probabilidade
4. SHAP explica quais fatores influenciaram a decisão
5. Médico analisa predição + explicação + contexto clínico completo
6. Médico toma decisão final considerando todos os fatores

### Limitações

O modelo foi treinado em dataset específico. Performance em dados reais pode variar. É necessário validação externa com dados do hospital antes de uso clínico.

Dataset possui viés de seleção (dados de pessoas que fizeram exames). Pode não representar população geral.

Modelo não considera fatores contextuais importantes como sintomas atuais, medicações em uso, ou outras condições médicas.

### Considerações Éticas

Monitoramento constante é necessário para detectar data drift (mudança na distribuição dos dados ao longo do tempo).

Análise de fairness deve verificar se modelo tem viés contra grupos demográficos específicos.

Transparência é fundamental: pacientes devem saber que IA está sendo usada e entender limitações.

---

## Próximos Passos

### Melhorias Técnicas

Grid Search ou Bayesian Optimization para otimizar hiperparâmetros. Ajuste de threshold para otimizar Recall em contexto médico. Ensemble de modelos para combinar diferentes perspectivas.

### Validação Clínica

Teste com dados reais do hospital. Validação por profissionais médicos das explicações SHAP. Estudo prospectivo comparando diagnóstico com e sem auxílio do sistema.

### Deploy

API REST para integração com sistemas hospitalares. Interface web para visualização de predições e explicações. Sistema de logging e monitoramento de performance em produção.

---

## Documentação Adicional

**docs/GUIA_INTERPRETACAO_RESULTADOS.md**
Explicação detalhada de todas as métricas e como interpretá-las no contexto médico.

**docs/RECOMENDACOES_MODELAGEM.md**
Discussão sobre escolha de features e modelos, incluindo análise de multicolinearidade.

**docs/README_PREPROCESSING.md**
Documentação completa do pipeline de pré-processamento e classe reutilizável.

---

## Licença

Este projeto foi desenvolvido como trabalho acadêmico para FIAP (Faculdade de Informática e Administração Paulista) no contexto do Tech Challenge da Pós-Tech em Inteligência Artificial para Desenvolvedores.

O código é disponibilizado para fins educacionais. Uso em produção requer validação clínica apropriada e aprovação regulatória.
