# 🏥 Sistema de Predição de Diabetes com Machine Learning

Um sistema abrangente de machine learning para auxiliar no diagnóstico precoce de diabetes mellitus, desenvolvido com foco na aplicação clínica e segurança do paciente.

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Metodologia](#-metodologia)
- [Modelos Implementados](#-modelos-implementados)
- [Métricas e Avaliação](#-métricas-e-avaliação)
- [Resultados](#-resultados)
- [Considerações Clínicas](#-considerações-clínicas)
- [Contribuição](#-contribuição)

## 🎯 Visão Geral

Este projeto implementa um sistema de machine learning para predição de diabetes usando indicadores de saúde. O sistema foi projetado especificamente para o contexto médico, priorizando:

- **Segurança do paciente**: Minimização de falsos negativos
- **Interpretabilidade**: Modelos explicáveis para uso clínico
- **Robustez**: Validação rigorosa e métricas médicas apropriadas
- **Praticidade**: Interface clara para profissionais de saúde

### 🎪 Problema Clínico

O diabetes mellitus afeta milhões de pessoas globalmente e frequentemente permanece não diagnosticado até o desenvolvimento de complicações. Este sistema visa:

- Identificar pacientes em risco de diabetes
- Apoiar triagem em larga escala
- Reduzir tempo para diagnóstico
- **Importante**: Complementar, não substituir, avaliação médica

## 📁 Estrutura do Projeto

```
diabetes-prediction/
├── src/                          # Módulos principais
│   ├── data_preprocessing.py     # Processamento e limpeza de dados
│   ├── model_training.py         # Treinamento de modelos ML
│   └── evaluation.py            # Avaliação e métricas médicas
├── notebooks/                    # Notebooks Jupyter
│   └── diabetes_analysis.ipynb  # Análise principal
├── outputs/                      # Resultados e visualizações
│   ├── models/                  # Modelos treinados salvos
│   ├── plots/                   # Gráficos e visualizações
│   └── reports/                 # Relatórios clínicos
├── diabetes_dataset.csv         # Dataset original
└── README.md                    # Este arquivo
```

## 🔧 Instalação

### Pré-requisitos

- Python 3.8+
- pip ou conda

### Instalação Rápida

1. **Clone o repositório**:

   ```bash
   git clone <repo-url>
   cd diabetes-prediction
   ```

2. **Instale as dependências**:

   ```bash
   pip install -r requirements.txt
   ```

   Ou crie um ambiente conda:

   ```bash
   conda create -n diabetes-ml python=3.9
   conda activate diabetes-ml
   pip install -r requirements.txt
   ```

### Dependências Principais

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
xgboost>=1.5.0  # Opcional
plotly>=5.0.0   # Opcional para gráficos interativos
```

## 🚀 Como Usar

### 1. Análise Completa (Recomendado)

Execute o notebook principal que utiliza todos os módulos:

```bash
jupyter notebook notebooks/diabetes_analysis.ipynb
```

### 2. Uso dos Módulos Individuais

#### Preprocessamento de Dados

```python
from src.data_preprocessing import DiabetesDataProcessor

# Inicializar processador
processor = DiabetesDataProcessor(n_features=15)

# Pipeline completo
results = processor.process_pipeline(
    filepath="diabetes_dataset.csv",
    test_size=0.2,
    val_size=0.2
)

X_train, X_val, X_test, y_train, y_val, y_test, exploration, outliers = results
```

#### Treinamento de Modelos

```python
from src.model_training import ModelTrainer

# Inicializar treinador
trainer = ModelTrainer(random_state=42)

# Treinar todos os modelos
results = trainer.train_all_models(
    X_train, y_train, X_val, y_val,
    tune_hyperparams=True
)

# Salvar modelos
trainer.save_models("outputs/models/")
```

#### Avaliação e Métricas

```python
from src.evaluation import MedicalModelEvaluator

# Inicializar avaliador
evaluator = MedicalModelEvaluator()

# Avaliar modelo específico
evaluator.evaluate_single_model(
    "random_forest", y_test, predictions, probabilities
)

# Comparar múltiplos modelos
comparison_df = evaluator.compare_models(results, "test")

# Gerar relatório clínico
clinical_report = evaluator.create_clinical_report("random_forest")
```

### 3. Pipeline Rápido

Para uma execução rápida do pipeline completo:

```python
# Executar tudo em uma única função
from notebooks.diabetes_analysis import run_full_pipeline

results = run_full_pipeline(
    data_path="diabetes_dataset.csv",
    save_outputs=True
)
```

## 🔬 Metodologia

### 1. Processamento de Dados

- **Exploração**: Análise estatística descritiva completa
- **Limpeza**: Tratamento de valores missing e outliers
- **Encoding**: Codificação de variáveis categóricas
- **Seleção**: SelectKBest com testes estatísticos
- **Normalização**: StandardScaler para todas as features
- **Divisão**: Estratificada 60%/20%/20% (treino/val/teste)

### 2. Validação

- **Validação cruzada**: 5-fold estratificada
- **Métricas médicas**: Foco em sensibilidade e especificidade
- **Otimização de limiares**: Baseada em índices clínicos
- **Teste independente**: Conjunto nunca visto pelos modelos

## 🤖 Modelos Implementados

| Modelo                  | Descrição             | Vantagens                   | Uso Clínico                      |
| ----------------------- | --------------------- | --------------------------- | -------------------------------- |
| **Regressão Logística** | Linear, interpretável | Coeficientes claros, rápido | ✅ Baseline, fácil interpretação |
| **Random Forest**       | Ensemble de árvores   | Robusto, feature importance | ✅ Boa performance geral         |
| **SVM**                 | Margem máxima         | Efetivo em alta dimensão    | ⚠️ Menos interpretável           |
| **KNN**                 | Baseado em vizinhança | Simples, não-paramétrico    | ⚠️ Sensível à escala             |
| **XGBoost**             | Gradient boosting     | Alta performance            | ✅ Estado da arte                |

## 📊 Métricas e Avaliação

### Métricas Médicas Principais

- **Sensibilidade (Recall)**: % de diabéticos identificados corretamente
- **Especificidade**: % de não-diabéticos identificados corretamente
- **PPV**: Probabilidade de diabetes quando teste positivo
- **NPV**: Probabilidade de não-diabetes quando teste negativo
- **F1-Score**: Métrica balanceada para dados desbalanceados

### Análise de Erros Clínicos

- **Falsos Negativos**: 🚨 **CRÍTICOS** - Pacientes diabéticos não detectados
- **Falsos Positivos**: ⚠️ Moderados - Encaminhamentos desnecessários

### Visualizações

- Matrizes de confusão interpretadas
- Curvas ROC e Precision-Recall
- Feature importance rankings
- Comparação de métricas entre modelos

## 📈 Resultados

> **Nota**: Os resultados específicos serão atualizados após execução completa do pipeline.

### Performance Geral (Exemplo)

| Modelo              | Accuracy | Sensibilidade | Especificidade | F1-Score | AUC-ROC |
| ------------------- | -------- | ------------- | -------------- | -------- | ------- |
| Random Forest       | 0.847    | 0.823         | 0.856          | 0.785    | 0.912   |
| XGBoost             | 0.851    | 0.819         | 0.862          | 0.788    | 0.918   |
| Logistic Regression | 0.834    | 0.798         | 0.847          | 0.761    | 0.891   |

### Features Mais Importantes

1. **Glucose** - Nível de glicose (correlação direta)
2. **BMI** - Índice de massa corporal
3. **Age** - Idade do paciente
4. **HighBP** - Pressão arterial elevada
5. **GenHlth** - Saúde geral auto-reportada

## ⚕️ Considerações Clínicas

### ✅ Uso Apropriado

- **Triagem populacional**: Identificação de pacientes em risco
- **Apoio diagnóstico**: Ferramenta complementar ao juízo clínico
- **Priorização**: Organização de filas de atendimento

### ⚠️ Limitações Importantes

- **NÃO substitui**: Avaliação médica completa
- **Supervisão obrigatória**: Profissional médico deve validar
- **Contexto específico**: Treinado em população americana
- **Viés potencial**: Verificar equidade em diferentes grupos

### 🔒 Aspectos Éticos

- **Transparência**: Explicar limitações aos pacientes
- **Consentimento**: Informar sobre uso de IA no diagnóstico
- **Responsabilidade**: Médico mantém decisão final
- **Privacidade**: Proteger dados de saúde sensíveis

## 📋 Requisitos do Sistema

### Hardware Mínimo

- RAM: 4GB (recomendado 8GB+)
- CPU: Dual-core
- Armazenamento: 1GB livre

### Software

- Python 3.8+
- Jupyter Notebook ou JupyterLab
- Navegador web moderno

## 🐛 Solução de Problemas

### Problemas Comuns

1. **Erro de importação XGBoost**:

   ```bash
   pip install xgboost
   ```

2. **Erro de memória**:

   - Reduzir `n_features` no preprocessor
   - Usar `n_jobs=1` nos modelos

3. **Gráficos não aparecem**:

   - Verificar `%matplotlib inline` no notebook
   - Instalar `plotly` para gráficos interativos

4. **Modelo não converge**:
   - Aumentar `max_iter` na Regressão Logística
   - Verificar normalização dos dados

## 👥 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. **Fork** o repositório
2. **Crie** uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. **Commit** suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. **Push** para a branch (`git push origin feature/nova-feature`)
5. **Abra** um Pull Request

### Áreas para Contribuição

- 🔬 Novos algoritmos de ML
- 📊 Visualizações adicionais
- 🏥 Métricas clínicas específicas
- 🌐 Interface web para uso clínico
- 📚 Documentação e tutoriais

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 📞 Contato

- **Autor**: [Seu Nome]
- **Email**: [seu.email@exemplo.com]
- **LinkedIn**: [seu-perfil-linkedin]

## 🙏 Agradecimentos

- Dataset: CDC Behavioral Risk Factor Surveillance System
- Comunidade Scikit-learn pela excelente documentação
- Profissionais de saúde consultados na validação clínica

---

> ⚠️ **Aviso Legal**: Este sistema é apenas para fins educacionais e de pesquisa. Não deve ser usado para diagnósticos médicos reais sem validação clínica apropriada e supervisão médica. Sempre consulte um profissional de saúde qualificado para questões médicas.

---

_Desenvolvido com ❤️ para melhorar o diagnóstico de diabetes através de IA responsável._
