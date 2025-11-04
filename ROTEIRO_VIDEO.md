# Roteiro - Vídeo de Apresentação (15 minutos)
## Sistema de Suporte ao Diagnóstico de Diabetes

---

## ESTRUTURA TEMPORAL

**Total:** 15 minutos
**Formato:** Screencast com narração + demonstração prática

---

## MINUTO 0:00 - 1:00 | INTRODUÇÃO (1 min)

### O que mostrar:
Tela inicial do projeto no VSCode/Jupyter

### O que falar:

"Olá, somos Hiago e Mylena, e este é nosso Tech Challenge da Fase 1. Desenvolvemos um sistema de suporte ao diagnóstico de diabetes usando Machine Learning."

"O desafio era criar uma ferramenta que ajude médicos a analisar exames de forma mais rápida e precisa. Importante: este é um sistema de SUPORTE. O médico sempre tem a palavra final."

"Vou mostrar todo o processo: desde a análise dos dados até o modelo final com explicações interpretáveis."

### Transição:
Abrir estrutura de pastas do projeto

---

## MINUTO 1:00 - 2:30 | PROBLEMA E DATASET (1.5 min)

### O que mostrar:
README.md ou slide com informações do dataset

### O que falar:

"Escolhemos o Diabetes Health Indicators Dataset do Kaggle, com 100 mil pacientes e 31 variáveis incluindo dados demográficos, exames laboratoriais e histórico médico."

"A variável alvo é diagnosed_diabetes: zero para sem diabetes, um para com diabetes. Temos features importantes como HbA1c, glicose, pressão arterial, IMC, idade, histórico familiar."

"O dataset está balanceado: 60% com diabetes e 40% sem, o que facilita o treinamento."

### Transição:
Abrir notebooks/analise-exploratoria.ipynb

---

## MINUTO 2:30 - 4:30 | ANÁLISE EXPLORATÓRIA (2 min)

### O que mostrar:
Executar células do notebook de EDA (ou mostrar já executado)

### O que falar:

"Primeira etapa: análise exploratória. Analisamos 23 variáveis numéricas com histogramas e boxplots."

**[Mostrar histograma de HbA1c]**
"Veja o HbA1c: distribuição mostra separação clara entre pacientes com e sem diabetes. Valores acima de 6.5 indicam diabetes."

**[Mostrar matriz de correlação]**
"Identificamos correlações fortes. HbA1c correlaciona 0.93 com glicose pós-prandial e 0.70 com glicose em jejum. Isso faz sentido clinicamente."

**[Scroll para estatísticas descritivas]**
"Calculamos média, mediana, quartis, identificamos outliers. Tudo documentado para entender os dados antes de modelar."

"Importante: discutimos correlação versus causalidade. Correlação não implica causa. Ice cream e afogamentos se correlacionam, mas um não causa o outro."

### Transição:
Abrir notebooks/preprocessing.ipynb

---

## MINUTO 4:30 - 6:30 | PRÉ-PROCESSAMENTO (2 min)

### O que mostrar:
Principais células do notebook de pré-processamento

### O que falar:

"Segunda etapa: pré-processamento. Pipeline completo de limpeza e transformação."

**[Mostrar verificação de dados]**
"Primeiro verificamos qualidade: zero valores ausentes, zero duplicatas. Dataset já vem limpo."

**[Mostrar One-Hot Encoding]**
"Aplicamos One-Hot Encoding nas variáveis categóricas: gênero, etnia, nível educacional, status de emprego, tabagismo. Isso transforma categorias em números que o modelo entende."

**[Mostrar StandardScaler]**
"Normalizamos variáveis numéricas com StandardScaler. Todas ficam com média zero e desvio padrão um. Isso é crucial para modelos lineares."

**[Mostrar remoção de features]**
"Removemos features problemáticas: diabetes_stage causa data leakage, pois já indica o diagnóstico. Removemos também features redundantes identificadas pela análise de VIF."

**[Mostrar train/test split]**
"Split estratificado 80/20. Mantém proporção de classes: 60/40 tanto no treino quanto no teste."

"Resultado: X_train com 80 mil exemplos, X_test com 20 mil. 37 features após limpeza e encoding."

### Transição:
Abrir notebooks/treinamento.ipynb

---

## MINUTO 6:30 - 9:30 | MODELAGEM E AVALIAÇÃO (3 min)

### O que mostrar:
Células de treinamento dos 3 modelos

### O que falar:

"Terceira etapa: modelagem. Treinamos três modelos."

**[Executar ou mostrar Logistic Regression]**
"Primeiro: Logistic Regression, nosso baseline. Modelo linear interpretável. Accuracy de 88.5%, ROC-AUC de 93.3%. Bom ponto de partida."

**[Mostrar feature importance LR]**
"Os coeficientes mostram que HbA1c tem maior impacto positivo, seguido por glicose. Isso alinha com conhecimento médico."

**[Executar ou mostrar Random Forest]**
"Segundo: Random Forest. Ensemble de 200 árvores. Accuracy sobe para 91%, Precision incrível de 99.98%. Praticamente elimina falsos positivos."

**[Executar ou mostrar XGBoost]**
"Terceiro: XGBoost, nosso modelo vencedor. Accuracy de 91%, ROC-AUC de 94.1%. Melhor balanço entre todas as métricas."

**[Mostrar tabela comparativa]**
"Comparação final: XGBoost vence em ROC-AUC, a métrica mais importante para classificação binária."

**[Mostrar Confusion Matrix]**
"Analisando a Confusion Matrix do XGBoost: 10.213 verdadeiros positivos, 7.988 verdadeiros negativos. Mas temos 1.787 falsos negativos. Isso significa 15% dos casos de diabetes não são detectados."

"Em contexto médico, falsos negativos são CRÍTICOS. Não detectar diabetes é perigoso. Falsos positivos apenas geram exames adicionais."

**[Mostrar curvas ROC]**
"As curvas ROC confirmam: todos os modelos estão acima do classificador aleatório. XGBoost tem a melhor área sob a curva."

### Transição:
Scroll para seção de validação cruzada

---

## MINUTO 9:30 - 10:30 | VALIDAÇÃO CRUZADA (1 min)

### O que mostrar:
Resultados do Cross-Validation

### O que falar:

"Para garantir robustez, aplicamos validação cruzada 5-fold estratificada."

**[Mostrar resultados CV]**
"O modelo é treinado e testado 5 vezes em diferentes subconjuntos. XGBoost mantém ROC-AUC médio de 94.1% com desvio padrão de apenas 0.3%. Isso confirma que o modelo é estável e não está overfitado."

"Diferença entre CV e Test Set menor que 2% indica excelente generalização. O modelo funciona bem em dados não vistos."

### Transição:
Scroll para seção SHAP

---

## MINUTO 10:30 - 13:00 | EXPLICABILIDADE COM SHAP (2.5 min)

### O que mostrar:
Análise SHAP completa

### O que falar:

"A parte mais importante: explicabilidade. Usamos SHAP para entender POR QUE o modelo faz cada predição."

"SHAP é baseado em teoria dos jogos. Calcula a contribuição de cada feature para a predição final."

**[Mostrar SHAP Summary Plot do XGBoost]**
"O Summary Plot mostra importância global. HbA1c no topo, como esperado. Cada ponto é um paciente. Vermelho significa valor alto, azul valor baixo."

"Veja HbA1c: pontos vermelhos à direita aumentam probabilidade de diabetes. Pontos azuis à esquerda diminuem. Exatamente o que esperamos clinicamente."

**[Mostrar SHAP Waterfall - paciente COM diabetes]**
"Agora uma predição individual. Este paciente tem diabetes. O Waterfall mostra passo a passo:"

"Base value é a predição média, 0.6. HbA1c alto adiciona 0.8 de probabilidade. Glicose postprandial adiciona 0.3. Idade adiciona 0.1. Predição final: 1.8, bem acima do threshold. Modelo prediz diabetes."

**[Mostrar SHAP Waterfall - paciente SEM diabetes]**
"Agora um paciente sem diabetes. Base value 0.6. HbA1c baixo subtrai 0.9. Glicose normal subtrai 0.4. Predição final: negativo 0.7. Modelo prediz sem diabetes."

**[Mostrar Dependence Plot]**
"Os Dependence Plots mostram relações não-lineares. Veja HbA1c: abaixo de 5.7, impacto negativo. Entre 5.7 e 6.5, zona de transição. Acima de 6.5, forte impacto positivo. Isso segue diretrizes médicas de diagnóstico de diabetes."

"SHAP é essencial porque permite que médicos validem as decisões do modelo. Se o modelo usa lógica incorreta, o médico detecta e não confia na predição."

### Transição:
Mostrar código ou terminal

---

## MINUTO 13:00 - 15:00 | APLICAÇÃO PRÁTICA E CONCLUSÕES (2 min)

### O que mostrar:
Tela final com resumo ou estrutura do projeto

### O que falar:

"Como isso funciona na prática?"

"Fluxo recomendado: Sistema recebe dados do paciente. Aplica pré-processamento. Modelo faz predição. SHAP explica. Médico analisa predição mais explicação mais contexto clínico completo. Médico toma decisão final."

"O sistema NÃO substitui o médico. É uma ferramenta de suporte. Pense como um segundo opinião muito rápida que processa milhares de casos similares."

**[Mostrar arquivos salvos]**
"Entregáveis: Três modelos treinados salvos em formato pickle. Pipeline de pré-processamento reutilizável. Documentação completa de interpretação de métricas."

"Resultados obtidos: Accuracy de 91%, ROC-AUC de 94%. Modelo detecta 85% dos casos de diabetes. Precision de 99% minimiza falsos positivos."

"Limitações importantes: Modelo treinado em dataset específico. Performance em dados reais pode variar. Necessário validação externa antes de uso clínico. Dataset tem viés de seleção, contém apenas pessoas que fizeram exames."

"Próximos passos: Otimização de hiperparâmetros com Grid Search. Ajuste de threshold para priorizar Recall. Validação com dados reais do hospital. Estudo prospectivo comparando diagnóstico com e sem sistema."

"Considerações éticas: Monitoramento de data drift. Análise de fairness para detectar viés demográfico. Transparência total com pacientes sobre uso de IA."

**[Mostrar estrutura final do projeto]**
"Projeto completo organizado: notebooks de EDA, pré-processamento e treinamento. Dados processados. Modelos salvos. Scripts reutilizáveis. Documentação técnica."

"Tudo versionado no Git, reproduzível e documentado para uso acadêmico e potencial aplicação clínica após validação apropriada."

"Obrigado pela atenção. Dúvidas, entrar em contato através da plataforma da FIAP."

---

## CHECKLIST DE REQUISITOS ATENDIDOS

Durante o vídeo, garantir que mencionar:

**Dataset:**
Diabetes Health Indicators do Kaggle, 100 mil exemplos

**Exploração:**
Estatísticas descritivas, visualizações, correlação

**Pré-processamento:**
Limpeza, encoding, normalização, análise de correlação

**Modelagem:**
3 modelos (Logistic Regression, Random Forest, XGBoost)

**Separação:**
Train/validation/test com validação cruzada

**Treinamento:**
Métricas completas (accuracy, precision, recall, F1, ROC-AUC)

**Interpretação:**
Feature importance E SHAP

**Discussão crítica:**
Limitações, aplicação prática, papel do médico

**Código:**
Projeto Python estruturado e documentado

**Organização:**
Notebooks, scripts, modelos, documentação

---

## DICAS DE GRAVAÇÃO

**Ritmo:**
Falar de forma clara mas não muito devagar. 15 minutos passam rápido.

**Demonstração:**
Preferir mostrar resultados já executados do que executar células (economiza tempo).

**Foco:**
Priorizar SHAP e aplicação prática (diferenciais do projeto).

**Erros:**
Se der erro, não gravar de novo. Explicar e seguir em frente.

**Ensaio:**
Fazer pelo menos um ensaio cronometrado antes da gravação final.

**Energia:**
Manter tom entusiasmado mas profissional. Este é um trabalho acadêmico sério.

---

## ARQUIVOS PARA TER ABERTOS

1. notebooks/analise-exploratoria.ipynb (células executadas)
2. notebooks/preprocessing.ipynb (células executadas)
3. notebooks/treinamento.ipynb (células executadas, especialmente SHAP)
4. README.md (para mostrar estrutura)
5. Pasta models/ e data/ (para mostrar arquivos gerados)

**Boa sorte na apresentação!**
